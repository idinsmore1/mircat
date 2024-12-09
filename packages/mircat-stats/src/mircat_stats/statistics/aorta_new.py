import numpy as np
import SimpleITK as sitk

from loguru import logger
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.centerline_new import Centerline
from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.segmentation import Aorta, SegNotFoundError


# This is the list of all vertebrae that could potentially show in specific regions of the aorta
AORTA_REGIONS_VERT_MAP: dict = {
    "abdominal": [*[f"L{i}" for i in range(1, 6)], "T12L1"],
    "thoracic": [f"T{i}" for i in range(1, 13)],
    "descending": [f"T{i}" for i in range(5, 13)],
}
# These are the default values for the aorta
AORTA_CROSS_SECTION_SPACING_MM: tuple = (1, 1)
AORTA_ROOT_LENGTH_MM: int = 10
AORTA_LABEL: int = 1
AORTA_ANISOTROPIC_SPACING_MM: tuple = (1, 1, 1)
AORTIC_CROSS_SECTION_SPACING_MM: tuple = (1, 1)
AORTIC_CROSS_SECTION_SIZE_MM: tuple = (100, 100)


@timer
def calculate_aorta_stats(nifti: MircatNifti) -> dict[str, float]:
    """Calculate the statistics for the aorta in the segmentation.
    Parameters:
    -----------
    nifti : MircatNifti
        The nifti obbject to calculate statistics for
    Returns:
    --------
    dict[str, float]
        The statistics for the aorta
    """
    # Filter to the segmentations we need
    try:
        aorta = Aorta(nifti)
    except SegNotFoundError as e:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return {}
    except Exception as e:
        logger.opt(exception=True).error(f"Error filtering to aorta in {nifti.path}")
        return {}
    # Calculate the aorta statistics
    aorta_stats = aorta.measure_statistics()