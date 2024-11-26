"""
This file contains the implementation of the NiftiMircat class, which represents a folder of NIfTI files output from MirCATo and keeps state and meta information.

The NiftiMircat class provides methods for setting up the metadata for the NIfTI file, checking if it is valid for specific tasks, loading the original CT and segmentation arrays, and loading statistics.

Classes:
- BodySegNotFoundError: Raised when the body segmentation file is not found.
- TotalSegNotFoundError: Raised when the total segmentation file is not found.
- TissuesSegNotFoundError: Raised when the tissues segmentation file is not found.
- NiftiMircat: A class to represent a folder of NIfTI files output from MirCATo and keep state information.

Functions:
- resample_nifti_sitk: Load and resample a NIfTI file using SimpleITK.
"""

import json
import SimpleITK as sitk
from pathlib import Path
from loguru import logger
from numpy import median

from mircat_stats.configs.statistics import stats_output_keys, midline_keys
from mircat_stats.configs.models import torch_model_configs
from mircat_stats.statistics.utils import _calc_shape_stats


class BodySegNotFoundError(FileNotFoundError):
    """
    Raised when the body segmentation file is not found
    """

    pass


class TotalSegNotFoundError(FileNotFoundError):
    """
    Raised when the total segmentation file is not found
    """

    pass


class TissuesSegNotFoundError(FileNotFoundError):
    """
    Raised when the tissues segmentation file is not found
    """

    pass


class MircatNifti:
    """
    A class to represent a folder of NIfTI files used by MirCAT for statistics.
    """

    task_labels = ["total", "tissues", "body"]

    def __init__(self, path: str):
        self.path = Path(path)  # assert that self.path is a Path object
        assert self.path.exists(), FileNotFoundError(f"Nifti {self.path} does not exist")

    def __str__(self):
        return str(self.path)

    def setup(self, task_list: list[str], gaussian: bool):
        """Setup the metadata for the NIfTI file and check if it is valid for tasks in task_list
        Parameters
        ----------
        task_list: list[str]
            A list of tasks to check for
        gaussian: bool
            Whether to apply Gaussian smoothing to the segmentation - slower but more precise.
        """
        # Set up the other meta information
        self.folder = self.path.parent
        self.nifti_name = self.path.name.partition(".")[0]
        self.seg_folder = self.folder.absolute() / f"{self.nifti_name}_segs"
        self.header_file = self.folder.absolute() / "header_info.json"
        assert self.seg_folder.exists(), FileNotFoundError(f"Segmentation folder {self.seg_folder} does not exist")
        # Set up the stat attributes
        self.output_file = self.seg_folder / f"{self.nifti_name}_stats.json"

        # Create a dictionary for segmentation files
        self.seg_files = {task: self.seg_folder / f"{self.nifti_name}_{task}.nii.gz" for task in self.task_labels}
        if self.output_file.exists():
            self._load_stats()
        else:
            self.stats_exist = False
            self.vert_midlines = {}
        self._load_original_ct()
        self._check_seg_files(task_list)
        self._load_seg_arrays(task_list, gaussian)
        self._check_and_load_header()
        # Calculate the vertebral midlines as reference points in the segmentations
        if self.vert_midlines == {}:
            self._get_vertebra_midlines(gaussian)

    def _check_and_load_header(self) -> None:
        if self.header_file.exists():
            try:
                with self.header_file.open("r") as f:
                    self.header_data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.warning("Header file {self.header_file} is corrupt or empty.")
                self.header_data = {}
        else:
            logger.warning(f"Header file {self.header_file} does not exist.")
            self.header_data = {}

        self.header_data["nifti_path"] = str(self.path.absolute())
        self.header_data["nii_file_name"] = str(self.nifti_name)

    def _check_seg_files(self, task_list: list[str]) -> None:
        """
        Check if the appropriate segmentation files exist for the given tasks
        Parameters
        ----------
        task_list: list[str]
            A list of tasks to check for
        """
        # Need total segmentation for vertebra midlines - if this doesn't exist it wont work
        if not self.seg_files["total"].exists():
            raise TotalSegNotFoundError(f'Total segmentation file {self.seg_files["total"]} does not exist')
        needs_tissues = "tissues" in task_list
        if needs_tissues:
            if not self.seg_files["body"].exists():
                raise BodySegNotFoundError(f'Body segmentation file {self.seg_files["body"]} does not exist')
            if not self.seg_files["tissues"].exists():
                raise TissuesSegNotFoundError(f'Tissues segmentation file {self.seg_files["tissues"]} does not exist')

    def _load_original_ct(self, resample_spacing=[1.0, 1.0, 1.0], gaussian=False) -> None:
        """Load in the original CT array"""
        self.original_ct = resample_nifti_sitk(self.path, resample_spacing, is_label=False, gaussian=gaussian)

    def _load_seg_arrays(self, task_list, gaussian: bool, resample_spacing=[1.0, 1.0, 1.0]) -> None:
        """Load in the original CT and segmentation arrays
        Parameters
        ----------
        task_list: list[str]
            The list of statistics tasks
        gaussian: bool
            Whether to apply Gaussian smoothing to the segmentation
        resample_spacing: list[float]
            The spacing for the resampling
        """
        needs_total = any([task in ["total", "aorta", "contrast"] for task in task_list])
        needs_tissues = "tissues" in task_list
        original_size = self.original_ct.GetSize()
        if needs_total:
            self.total_seg = resample_nifti_sitk(
                self.seg_files["total"],
                resample_spacing,
                is_label=True,
                gaussian=gaussian,
            )
            if original_size != self.total_seg.GetSize():
                raise ValueError("Size of total segmentation is not consistent with original ct.")
        if needs_tissues:
            self.body_seg = resample_nifti_sitk(
                self.seg_files["body"],
                resample_spacing,
                is_label=True,
                gaussian=gaussian,
            )
            self.tissues_seg = resample_nifti_sitk(
                self.seg_files["tissues"],
                resample_spacing,
                is_label=True,
                gaussian=gaussian,
            )
            if original_size != self.body_seg.GetSize():
                raise ValueError("Size of body segmentation is not consistent with original ct.")
            if original_size != self.tissues_seg.GetSize():
                raise ValueError("Size of tissues segmentation is not consistent with original ct.")

    def _load_stats(self) -> None:
        try:
            with self.output_file.open("r") as f:
                stats = json.load(f)
            self.stats = stats
            vert_midlines = {k: stats.get(k) for k in midline_keys}
            self.vert_midlines = {k: v for k, v in vert_midlines.items() if v is not None}
            if self.vert_midlines.get("vertebrae_T12L1_midline") is None:
                if (
                    self.vert_midlines.get("vertebrae_L1_midline") is not None
                    and self.vert_midlines.get("vertebrae_T12_midline") is not None
                ):
                    self.vert_midlines["vertebrae_T12L1_midline"] = int(
                        (self.vert_midlines["vertebrae_L1_midline"] + self.vert_midlines["vertebrae_T12_midline"]) / 2
                    )
            self.stats_exist = True

        except json.decoder.JSONDecodeError:
            logger.warning(f"Stats file {self.output_file} is corrupt or empty.")
            self.stats = {}
            self.stats_exist = False
            self.vert_midlines = {}

    def _get_vertebra_midlines(self, gaussian: bool) -> None:
        # Calculate the vertebral midlines
        total_map = torch_model_configs["total"]["output_map"]
        vert_midlines = {}
        if getattr(self, "total_seg", False):
            total_seg = self.total_seg
        else:
            self._load_seg_arrays(["total"], gaussian)
            total_seg = self.total_seg
        shape_stats = _calc_shape_stats(total_seg)
        seg_labels = shape_stats.GetLabels()
        vert_map = {k: v for k, v in total_map.items() if "vertebrae" in k}
        for name, label in vert_map.items():
            if label in seg_labels:
                indices = shape_stats.GetIndexes(label)
                z_indices = indices[2::3]
                vert_midlines[f"{name}_midline"] = int(median(z_indices))
        if "vertebrae_L1_midline" in vert_midlines and "vertebrae_T12_midline" in vert_midlines:
            vert_midlines["vertebrae_T12L1_midline"] = int(
                (vert_midlines["vertebrae_L1_midline"] + vert_midlines["vertebrae_T12_midline"]) / 2
            )
        self.vert_midlines = vert_midlines

    def write_stats_to_file(self, output_stats: dict, all_completed: bool) -> None:
        """Write the statistics to a JSON file
        Parameters
        ----------
        output_stats: dict
            The statistics to write
        all_completed: bool
            Whether all statistics have been completed
        """
        if self.stats_exist:
            existing_stats = self.stats
            existing_stats.update(output_stats)
            output_stats = existing_stats.copy()
        if all_completed:
            flag_file = str(self.output_file).replace("_stats.json", ".complete")
            Path(flag_file).touch()
        output_stats = {k: output_stats.get(k) for k in stats_output_keys}
        with self.output_file.open("w") as f:
            json.dump(output_stats, f, indent=4)


def resample_nifti_sitk(nifti_path: Path, new_spacing: list[float], is_label: bool, gaussian: bool) -> sitk.Image:
    """Load and resample a NIfTI file using SimpleITK
    Parameters
    ----------
    nifti_path: Path
        The path to the NIfTI file
    new_spacing: list[float]
        The new spacing for the resampling
    is_label: bool
        Whether the NIfTI file is a label map
    gaussian: bool
        Whether to apply Gaussian smoothing to the image

    Returns
    -------
    sitk.Image
        The resampled image
    """
    input_image = sitk.ReadImage(nifti_path)
    # Get the original spacing and size
    return _resample(input_image, new_spacing, is_label, gaussian)


def _resample(input_image, new_spacing, is_label, gaussian):
    original_spacing = input_image.GetSpacing()
    original_size = input_image.GetSize()

    # Calculate the new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]
    if is_label:
        if gaussian:
            interpolator = sitk.sitkLabelGaussian
        else:
            interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    # interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    # Create the resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_image.GetPixelIDValue())

    resample.SetInterpolator(interpolator)

    # Perform the resampling
    return resample.Execute(input_image)
