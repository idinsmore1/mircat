import numpy as np

from loguru import logger
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.segmentation import Segmentation, SegNotFoundError


# This is the list of all vertebrae that could potentially show in specific regions of the aorta
AORTA_REGIONS_VERT_MAP: dict = {
    "abdominal": [*[f"L{i}" for i in range(1, 6)], "T12L1"],
    "thoracic": [f"T{i}" for i in range(1, 13)],
    "descending": [f"T{i}" for i in range(5, 13)],
}
AORTA_CROSS_SECTION_SPACING_MM: tuple = (1, 1)
AORTA_ROOT_LENGTH_MM: int = 10


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
    aorta_stats: dict = {}
    vert_midlines: dict = nifti.vert_midlines
    thoracic, abdominal, descending = check_aortic_regions_in_segmentation(vert_midlines)
    if not any([thoracic, abdominal, descending]):
        logger.warning(f"No aortic regions found in {nifti.path}")
        return aorta_stats
    # Filter to the segmentations we need
    try:
        aorta = Segmentation(nifti, ['aorta', 'brachiocephalic_trunk', 'subclavian_artery_left'])
        aorta_seg = aorta.segmentation
        aorta_img = aorta.original_ct
    except SegNotFoundError as e:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return aorta_stats
    except Exception as e:
        logger.opt(exception=True).error(f"Error filtering to aorta in {nifti.path}")
        return aorta_stats
    if abdominal:
        start, end = find_aortic_region_endpoints("abdominal", vert_midlines)
    if thoracic:
        start, end = find_aortic_region_endpoints("thoracic", vert_midlines)
    elif descending:  # Descending aorta only is run independently if ascending is not present
        start, end = find_aortic_region_endpoints("descending", vert_midlines)


def check_aortic_regions_in_segmentation(vert_midlines: dict) -> tuple[bool, bool, bool]:
    """Check which aortic regions are in the segmentation.
    Parameters:
    -----------
    vert_midlines : dict
        The vertebral midlines of the segmentation
    Returns:
    --------
    tuple[bool, bool, bool]
        Whether the thoracic, abdominal, and descending aorta are in the segmentation
    """
    # T4 and at least T8 need to be in the image for thoracic to be measured
    thoracic = vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get("vertebrae_T4_midline", False)
    # L3 has to be in the image for abdominal to be measured
    abdominal = vert_midlines.get("vertebrae_L3_midline", False)
    # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
    descending = vert_midlines.get("vertebrae_T12_midline", False) and vert_midlines.get("vertebrae_T9_midline", False)
    return thoracic, abdominal, descending


def find_aortic_region_endpoints(region: str, vert_midlines: dict) -> tuple[int, int]:
    """Find the endpoints of the aortic region in the segmentation.
    Parameters:
    -----------
    region : str
        The region of the aorta to find the endpoints for
    vert_midlines : dict
        The vertebral midlines of the segmentation
    Returns:
    --------
    tuple[int, int]
        The [start, end] of the aorta region (inclusive on both sides)
    """
    possible_locs = AORTA_REGIONS_VERT_MAP[region]
    midlines = [
        vert_midlines.get(f'vertebrae_{vert}_midline')
        for vert in possible_locs
        if vert_midlines.get(f'vertebrae_{vert}_midline') is not None
    ]
    midlines = [midline for midline in midlines if midline]
    start = min(midlines)
    end = max(midlines) + 1  # add one to make it inclusive
    return start, end

