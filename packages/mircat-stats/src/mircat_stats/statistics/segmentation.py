import SimpleITK as sitk
import numpy as np
from loguru import logger

from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.utils import _filter_largest_components
from mircat_stats.configs.models import torch_model_configs


class SegNotFoundError(ValueError):
    """
    Raised when the aorta segmentation is not found
    """

    pass


class ArchNotFoundError(ValueError):
    """
    Raised when the aortic arch can not be defined
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
        self.vert_midlines = nifti.vert_midlines.copy()
        self.seg_folder = nifti.seg_folder
        self.nifti_name = nifti.nifti_name
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
        return self

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
        return self

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
        self.bbox = bbox
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
        self.cropped_seg = cropped_seg
        self.cropped_ct = cropped_img
        return self

    def _make_SPR_numpy_array(self, on_cropped: bool=False) -> None:
        """Convert the images to numpy arrays with with axes in SPR (S, P, R) format. Superior (0) -> Inferior (N - 1), Anterior -> Posterior, Left -> Right
        Parameters
        ----------
        on_cropped : bool
            Use the cropped bounding box image for the conversion instead of the full image
        """
        if on_cropped:
            assert hasattr(self, "cropped_seg"), ValueError("Cropped segmentation must be extracted before conversion")
            assert hasattr(self, "cropped_ct"), ValueError("Cropped CT image must be extracted before conversion")
            seg = self.cropped_seg
            ct = self.cropped_ct
        else:
            seg = self.segmentation
            ct = self.original_ct

        self.segmentation_arr = np.flip(np.flipud(sitk.GetArrayFromImage(seg)), axis=1)
        self.original_ct_arr = np.flip(np.flipud(sitk.GetArrayFromImage(ct)), axis=1)
        # Adjust the vertebral midlines to account for the flip
        new_midlines = {k: (self.segmentation_arr.shape[0] - 1) - v for k, v in self.vert_midlines.items()}
        self.vert_midlines = new_midlines
