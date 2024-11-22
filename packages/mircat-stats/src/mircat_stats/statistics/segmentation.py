import SimpleITK as sitk
import numpy as np
from loguru import logger

from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.utils import _filter_largest_components
from mircat_stats.configs.models import torch_model_configs

class Segmentation:
    """Class to filter one or multiple segmentations out from a single model and 
    hold them in a single object. This is useful for specific morphology-based statistics
    """
    def __init__(self, nifti: MircatNifti, seg_names: list[str]):
        self.original_ct = nifti.original_ct
        self.seg_names = seg_names
        self._find_seg_models()
        self._filter_to_segmentations()

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
    
    def _filter_to_segmentations(self, nifti: MircatNifti) -> sitk.Image:
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
        label_indices = [v['idx'] for v in self.seg_info.values()]
        label_map = {old_idx: new_idx for old_idx, new_idx in enumerate(label_indices, start=1)}
        # Make the image an array
        seg_arr = sitk.GetArrayFromImage(complete).astype(np.uint8)
        mask = np.isin(seg_arr, label_indices)
        seg_arr[~mask] = 0
        for old_idx, new_idx in label_map.items():
            seg_arr[seg_arr == old_idx] = new_idx
        mapped_indices = [int(x) for x in np.unique(seg_arr) if x != 0]
        if not mapped_indices:
            logger.opt(exception=True).error("No segmentations found in the input")
            raise ValueError("No segmentations found in the input")
        if set(mapped_indices) != set(label_map.values()):
            missing = ','.join([labels[x] for x in set(label_map.values()) - set(mapped_indices)])
            logger.warning(f'{missing} were not found in the segmentation')
        segmentation = sitk.GetImageFromArray(seg_arr)
        segmentation.CopyInformation(complete)
        self.segmentations = _filter_largest_components(segmentation, mapped_indices)
            



        

        