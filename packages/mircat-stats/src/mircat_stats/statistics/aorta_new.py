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
    aorta_stats: dict[str, float] = {}
    # Filter to the segmentations we need
    try:
        aorta = Aorta(nifti)
    except SegNotFoundError as e:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return aorta_stats
    except Exception as e:
        logger.opt(exception=True).error(f"Error filtering to aorta in {nifti.path}")
        return aorta_stats
    
    if not aorta.region_existence:
        return aorta_stats

    # Go through each region and measure it
    for region, has_region in region_existence.items():
        # If the region is not in the segmentation, skip it
        if not has_region:
            continue
        # If the region is descending and full thoracic is present, skip it
        if region == "descending" and region_existence["thoracic"]:
            continue
        # Find the start and end of the region
        try:
            start, end = _find_aortic_region_endpoints(region, vert_midlines)
            # Convert the segmentation and image to numpy arrays with the arch at the top
            aorta_seg_arr = _make_aorta_superior_array(aorta_seg[:, :, start:end])
            aorta_img_arr = _make_aorta_superior_array(aorta_img[:, :, start:end])
            # We clip the image houndsfield units to be between -200 and 250 for fat analysis
            aorta_img_arr.clip(-200, 250, out=aorta_img_arr)

        except Exception as e:
            logger.opt(exception=True).error(f"Error measuring {region} aorta in {nifti.path}")
            continue


def _check_aortic_regions_in_segmentation(vert_midlines: dict) -> dict[str, bool]:
    """Check which aortic regions are in the segmentation.
    Parameters:
    -----------
    vert_midlines : dict
        The vertebral midlines of the segmentation
    Returns:
    --------
    dict[str, bool]
        Whether the thoracic, abdominal, and descending aorta are in the segmentation
    """
    # T4 and at least T8 need to be in the image for thoracic to be measured
    thoracic = vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get("vertebrae_T4_midline", False)
    # L3 has to be in the image for abdominal to be measured
    abdominal = vert_midlines.get("vertebrae_L3_midline", False)
    # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
    descending = vert_midlines.get("vertebrae_T12_midline", False) and vert_midlines.get("vertebrae_T9_midline", False)
    return {"abdominal": abdominal, "descending": descending, "thoracic": thoracic}


def _find_aortic_region_endpoints(region: str, vert_midlines: dict) -> tuple[int, int]:
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
        vert_midlines.get(f"vertebrae_{vert}_midline")
        for vert in possible_locs
        if vert_midlines.get(f"vertebrae_{vert}_midline") is not None
    ]
    midlines = [midline for midline in midlines if midline]
    start = min(midlines)
    end = max(midlines) + 1  # add one to make it inclusive
    return start, end


def _make_aorta_superior_array(img: sitk.Image) -> np.ndarray:
    """
    Transform an sitk.Image to a numpy array and reorient so that the aortic arch is at the top of the image.
    Parameters
    ----------
    img : sitk.Image
        The sitk image to transform
    Returns
    -------
    np.array
        the reoriented numpy array
    """
    arr = sitk.GetArrayFromImage(img)
    arr = arr.transpose(2, 1, 0)
    arr = np.flip(np.rot90(arr, axes=(0, 2)), axis=1)
    arr = np.flip(arr, 1)
    return arr


def measure_aortic_region(seg_arr: np.ndarray, img_arr: np.ndarray, region: str) -> dict[str, float]:
    """Measure the aortic region in the segmentation.
    Parameters:
    -----------
    seg_arr : np.ndarray
        The segmentation array
    img_arr : np.ndarray
        The image array
    region : str
        The region of the aorta to measure
    Returns:
    --------
    dict[str, float]
        The measurements for the aorta region
    """
    region_stats = {}
    # Create the centerline from the segmentation
    centerline = Centerline(AORTA_ANISOTROPIC_SPACING_MM, AORTA_LABEL)
    centerline.create_centerline(seg_arr)
    if not centerline.succeeded:
        logger.warning(f"Failed to create centerline for {region} aorta")
        return region_stats
