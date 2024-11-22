import numpy as np

from mircat_stats.configs.models import torch_model_configs
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.utils import _calc_intensity_stats, _calc_shape_stats


@timer
def calculate_total_segmentation_stats(
    nifti: MircatNifti,
) -> tuple[dict[str:float], dict[str:int]]:
    """Calculate the statistics for the total segmentation
    Parameters
    ----------
    nifti : MircatNifti
        The nifti file to calculate statistics for
    Returns
    -------
    dict[str: float]
        The statistics for the total segmentation
    dict[str: int]
        The the vertebral midlines calculated from the total segmentation
    """
    # Set up the two output maps
    total_map = torch_model_configs["total"]["output_map"]
    # Calculate the stats for the total segmentation
    shape_stats = _calc_shape_stats(nifti.total_seg)
    intensity_stats = _calc_intensity_stats(nifti.original_ct, nifti.total_seg)
    seg_labels = shape_stats.GetLabels()
    total_stats = {}
    # Loop through the shape and intensity stats for each structure
    for name, label in total_map.items():
        if label in seg_labels:
            volume = round(shape_stats.GetPhysicalSize(label) / 1000, 1)
            intensity = round(intensity_stats.GetMean(label), 2)
            total_stats[f"{name}_volume_cm3"] = volume
            total_stats[f"{name}_average_intensity"] = intensity
    return total_stats
