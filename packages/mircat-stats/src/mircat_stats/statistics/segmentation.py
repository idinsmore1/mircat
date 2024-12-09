import SimpleITK as sitk
import numpy as np
from loguru import logger

from mircat_stats.statistics.nifti import MircatNifti, _resample
from mircat_stats.statistics.utils import _filter_largest_components
from mircat_stats.configs.models import torch_model_configs


class SegNotFoundError(ValueError):
    """
    Raised when the aorta segmentation is not found
    """

    pass


class SegmentationSizeDoesNotMatchError(ValueError):
    """
    Raised when the segmentation and CT image do not have the same size
    """

    pass


class Segmentation:
    """Class to filter one or multiple segmentations out from a single model and
    hold them in a single object. This is useful for specific morphology-based statistics
    """

    def __init__(self, nifti: MircatNifti, seg_names: list[str]):
        """Initialize Segmentation class.

        This class handles filtering and potentially analysis of segmented CT images.
        It will load and filter the appropriate complete segmentation on initialization.

        Args:
            nifti (MircatNifti): A MircatNifti object containing CT and segmentation data
            seg_names (list[str]): List of segmentation names to analyze

        Attributes:
            original_ct: Original CT image data
            vert_midlines: Vertebrae midline data
            seg_folder: Folder containing segmentation files
            seg_info: Dictionary containing segmentation information
            model: Model used for segmentation
            segmentation: Filtered segmentation image
            seg_names: List of segmentation names in output
        """
        self.path = nifti.path
        self.original_ct = nifti.original_ct
        self.vert_midlines = nifti.vert_midlines
        self.seg_folder = nifti.seg_folder
        self.seg_names = seg_names
        self._find_seg_model()
        self._filter_to_segmentation(nifti)

    def _find_seg_model(self):
        seg_info = {}
        for seg_name in self.seg_names:
            for model in torch_model_configs:
                if seg_name in torch_model_configs[model]["output_map"]:
                    seg_model = model
                    seg_idx = torch_model_configs[model]["output_map"][seg_name]
                    break
            seg_info[seg_name] = {"model": seg_model, "idx": seg_idx}
        model = set(info["model"] for info in seg_info.values())
        if len(model) > 1:
            raise ValueError("All segmentations must come from the same model")
        model = model.pop()
        self.seg_info = seg_info
        self.model = model

    def _filter_to_segmentation(self, nifti: MircatNifti) -> sitk.Image:
        """Filter input nifti to segmented regions.

        This method applies filtering to convert a nifti image into segmented regions.
        Labels will be indexed from 1 to len(labels).

        Args:
            nifti (MircatNifti): Input nifti image to be segmented.

        Returns:
            sitk.Image: Filtered image containing segmented regions.
        """
        if self.model == "total":
            complete = nifti.total_seg
        elif self.model == "body":
            complete = nifti.body_seg
        elif self.model == "tissues":
            complete = nifti.tissues_seg
        labels = list(self.seg_info.keys())
        label_indices = [v["idx"] for v in self.seg_info.values()]

        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(label_indices, start=1)}
        seg_arr = sitk.GetArrayFromImage(complete).astype(np.uint8)
        mask = np.isin(seg_arr, label_indices)
        seg_arr[~mask] = 0
        for old_idx, new_idx in label_map.items():
            seg_arr[seg_arr == old_idx] = new_idx
        mapped_indices = [int(x) for x in np.unique(seg_arr) if x != 0]
        if 1 not in mapped_indices:
            logger.opt(exception=True).error("No segmentations found in the input")
            raise SegNotFoundError("No segmentations found in the input")
        if set(mapped_indices) != set(label_map.values()):
            missing = set(label_map.values()).difference(set(mapped_indices))
            missing_labels = ",".join([labels[idx - 1] for idx in missing])
            logger.debug(f"{missing_labels} not found in the input")
            labels = [labels[idx - 1] for idx in mapped_indices]
        segmentation = sitk.GetImageFromArray(seg_arr)
        segmentation.CopyInformation(complete)
        self.segmentation = _filter_largest_components(segmentation, mapped_indices)
        self.seg_names = labels

    def extract_segmentation_bounding_box(self, padding: tuple[int] | int = (0, 0, 0)) -> tuple[sitk.Image, sitk.Image]:
        """
        Extract the bounding box of the segmentation with a given amount of padding around it.
        Args
        ----
        padding: tuple[int] | int
            The padding to add around the bounding box. If a single integer is given, the same padding will be added in all directions.
        Returns
        -------
        tuple[sitk.Image, sitk.Image]
            The cropped segmentation and CT image
        """
        assert self.segmentation.GetSize() == self.original_ct.GetSize(), SegmentationSizeDoesNotMatchError(
            "Segmentation and CT image must have the same size"
        )
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        else:
            assert len(padding) == 3, ValueError(
                "Bounding box padding must be a single integer or a tuple of 3 integers"
            )

        # Set up sitk filter
        bbox_filter = sitk.LabelShapeStatisticsImageFilter()
        bbox_filter.SetComputeOrientedBoundingBox(True)
        bbox_filter.Execute(self.segmentation)
        bbox = bbox_filter.GetBoundingBox(1)

        # Set up the cropping filter
        start_idx = list(bbox[0:3])
        size = list(bbox[3:6])
        for i in range(3):
            # Adjust start index
            start_idx[i] = max(0, start_idx[i] - padding[i])
            # Adjust size to account for padding and image bounds
            max_size = self.segmentation.GetSize()[i] - start_idx[i]
            size[i] = min(size[i] + 2 * padding[i], max_size)
        # Extract regions using the same coordinates for both images
        extract = sitk.ExtractImageFilter()
        extract.SetSize(size)
        extract.SetIndex(start_idx)

        # Extract from segmentation
        cropped_seg = extract.Execute(self.segmentation)
        cropped_img = extract.Execute(self.original_ct)
        return cropped_seg, cropped_img


class Aorta(Segmentation):
    # This is the list of all vertebrae that could potentially show in specific regions of the aorta
    vertebral_regions_map: dict = {
        "abdominal": [*[f"L{i}" for i in range(1, 6)], "T12L1"],
        "thoracic": [f"T{i}" for i in range(3, 13)],
        "descending": [f"T{i}" for i in range(5, 13)],
    }
    # These are the default values for the aorta
    cross_section_spacing_mm: tuple = (1, 1)
    root_length_mm: int = 10
    anisotropic_spacing_mm: tuple = (1, 1, 1)
    cross_section_size_mm: tuple = (100, 100)

    def __init__(self, nifti: MircatNifti):
        super().__init__(nifti, ["aorta", "brachiocephalic_trunk", "subclavian_artery_left"])
        self._check_aortic_regions_in_segmentation()
        self._make_aorta_superior_numpy_array()
        self._get_region_endpoints()

    #### INITIALIZATION OPERATIONS 
    def _check_aortic_regions_in_segmentation(self) -> dict[str, bool]:
        """Check if the aortic regions are present in the segmentation.
        Returns:
        --------
        dict[str, bool]
            Dictionary with keys as region names and values as boolean indicating if the region is present
        """
        vert_midlines = self.vert_midlines
            # T4 and at least T8 need to be in the image for thoracic to be measured
        thoracic = bool(vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get("vertebrae_T4_midline", False))
        # L3 has to be in the image for abdominal to be measured
        abdominal = bool(vert_midlines.get("vertebrae_L3_midline", False))
        # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
        descending = bool(vert_midlines.get("vertebrae_T12_midline", False) and vert_midlines.get("vertebrae_T9_midline", False))
        region_existence = {
            "thoracic": {
                'exists': thoracic, 
            },
            "abdominal": {
                'exists': abdominal, 
            },
            "descending": {
                'exists': descending, 
            }
        }
        self.region_existence = region_existence
        if not any([region_existence[region]['exists'] for region in region_existence]):
            raise SegNotFoundError(f"No aortic regions found in {self.path}")
        return self

    def _make_aorta_superior_numpy_array(self) -> None:
        """Convert the aorta segmentation to a numpy array with the arch at the top and adjust vertebral midlines"""
        self.segmentation_arr = np.flipud(sitk.GetArrayFromImage(self.segmentation))
        self.original_ct_arr = np.flipud(sitk.GetArrayFromImage(self.original_ct))
        # Clip the houndsfield units for fat analysis
        self.original_ct_arr = np.clip(self.original_ct_arr, -200, 250)
        # Adjust the vertebral midlines to account for the flip
        new_midlines = {k: (self.segmentation_arr.shape[0] - 1) - v for k, v in self.vert_midlines.items()}
        self.vert_midlines = new_midlines

    def _get_region_endpoints(self) -> None:
        for region, has_region in self.region_existence.items():
            if has_region:
                endpoints = self._find_aortic_region_endpoints(region, self.vert_midlines)
                self.region_existence[region]['endpoints'] = endpoints
    
    @staticmethod
    def _find_aortic_region_endpoints(region: str, vert_midlines: dict) -> tuple[int, int]:
        possible_locs = Aorta.vertebral_regions_map[region]
        midlines = [
            vert_midlines.get(f"vertebrae_{vert}_midline")
            for vert in possible_locs
            if vert_midlines.get(f"vertebrae_{vert}_midline") is not None
        ]
        midlines = [midline for midline in midlines if midline]
        start = min(midlines)
        end = max(midlines) + 1  # add one to make it inclusive
        return start, end


