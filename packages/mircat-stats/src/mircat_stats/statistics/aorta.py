import SimpleITK as sitk
import numpy as np

from loguru import logger
from functools import partial

from mircat_stats.configs.models import torch_model_configs
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.nifti import NiftiMircato
from mircat_stats.statistics.utils import _filter_largest_components
from mircat_stats.statistics.centerline import create_centerline
from mircat_stats.statistics.cpr import (
    create_straightened_cpr,
    measure_largest_cpr_diameter,
    measure_mid_cpr_diameter,
    _get_regions,
    measure_cross_sectional_diameter,
)


AORTA_CROSS_SECTION_SPACING = (1, 1)
ROOT_LENGTH = 10


@timer
def calculate_aorta_stats(nifti: NiftiMircato, vert_midlines: dict) -> dict:
    """Calculate the statistics for the aorta
    Parameters
    ----------
    nifti : NiftiMircato
        The nifti file to calculate statistics for
    vert_midlines : dict
        The vertebral midlines
    Returns
    -------
    dict[str: float]
        The statistics for the aorta
    """
    aorta_stats = {}
    # These are all the possible vertebral regions of the aorta. Scans will have some subset of these
    region_vert_map = {
        "abdominal": ["S1", *[f"L{i}" for i in range(1, 6)]],
        "thoracic": [f"T{i}" for i in range(1, 13)],
        "descending": [f"T{i}" for i in range(5, 13)],
    }
    # Define the output map
    # output_map = torch_model_configs["total"]["output_map"]
    # Get the regions to measure
    thoracic, abdominal, descending = _check_aorta_regions(vert_midlines)
    if not any((thoracic, abdominal, descending)):
        logger.warning("No aorta regions to measure")
        return aorta_stats
    # Get stats for each region of the aorta
    img = nifti.original_ct
    try:
        seg = _filter_to_aorta_seg(nifti.total_seg)
        measure_aorta_region = partial(_measure_aorta_region, img, seg)
        find_region_endpoints = partial(
            _find_aortic_region_endpoints, vert_midlines=vert_midlines
        )
    except Exception as e:
        logger.error(f"Error filtering to aorta segmentation: {e}")
        return aorta_stats

    if abdominal:
        try:
            start, end = find_region_endpoints(region_vert_map["abdominal"])
            abd_stats = measure_aorta_region(
                start, end, is_thoracic=False, region="abd_aorta"
            )
            aorta_stats.update(abd_stats)
        except Exception as e:
            logger.error(f"Error measuring abdominal aorta: {e}")
    if thoracic:
        try:
            start, end = find_region_endpoints(region_vert_map["thoracic"])
            thor_stats = measure_aorta_region(start, end, is_thoracic=True)
            aorta_stats.update(thor_stats)
        except Exception as e:
            logger.error(f"Error measuring thoracic aorta: {e}")
    elif descending:
        try:
            start, end = find_region_endpoints(region_vert_map["descending"])
            desc_stats = measure_aorta_region(
                start, end, is_thoracic=False, region="desc_aorta"
            )
            aorta_stats.update(desc_stats)
        except Exception as e:
            logger.error(f"Error measuring descending aorta: {e}")
    return aorta_stats


def _measure_aorta_region(
    img: sitk.Image,
    seg: sitk.Image,
    start: int,
    end: int,
    is_thoracic: bool,
    region: str | None = None,
) -> dict[str, float]:
    """Measure the aorta region in the image
    Parameters
    ----------
    img : sitk.Image
        The original image
    seg : sitk.Image
        The segmentation of the image
    start : int
        The start of the region to measure
    end : int
        The end of the region to measure
    is_thoracic : bool
        If the region is thoracic
    region : str
        The region to measure
    Returns
    -------
    dict[str: float]
        The statistics for the aorta region
    """
    # Get the region of the image to measure
    region_stats = {}
    # region_img = _aorta_superior(sitk.GetArrayFromImage(img[:, :, start:end]))
    region_seg = _aorta_superior(sitk.GetArrayFromImage(seg[:, :, start:end]))
    # Get the aorta segmentation
    aorta_label = 1
    aorta_anisotropy = (1, 1, 1)
    cross_section_size = (100, 100)
    cross_section_spacing = (1, 1)
    # Measure the diameters through a CPR
    centerline = create_centerline(
        region_seg, aorta_anisotropy, aorta_label, is_thoracic=is_thoracic
    )
    if centerline is None:
        logger.warning("No centerline found")
        return region_stats
    # Create the CPR
    cpr = create_straightened_cpr(
        region_seg, centerline, cross_section_xy=cross_section_size
    )
    if is_thoracic:
        diam_data = _measure_thoracic_diameters(region_seg, cpr, centerline)
        region_stats.update(diam_data)
    else:
        cpr = cpr[1:]  # remove the first slice to ignore oblong starting slices
        diam_data = measure_largest_cpr_diameter(cpr, cross_section_spacing)
        mid_diam = measure_mid_cpr_diameter(cpr, cross_section_spacing)
        diam_data.update(mid_diam)
        diam_data = {f"{region}_{k}": v for k, v in diam_data.items()}
        region_stats.update(diam_data)
    return region_stats


def _aorta_superior(arr: np.ndarray) -> np.ndarray:
    """Transform an array generated from a sitk image to the so that the arch of the aorta is at the top
    Parameters
    ----------
    arr : np.array
        The sitk -> numpy array to transform
    Returns
    -------
    np.array
        The LAS oriented array
    """
    arr = arr.transpose(2, 1, 0)
    arr = np.flip(np.rot90(arr, axes=(0, 2)), axis=1)
    arr = np.flip(arr, 1)
    return arr


def _filter_to_aorta_seg(total_seg: sitk.Image) -> sitk.Image:
    """Transform the total segmentation to only include the aorta and arch defining segmentations
    as well as perform morphological operations to properly orient the aorta
    Parameters
    ----------
    total_seg : sitk.Image
        The total segmentation
    Returns
    -------
    sitk.Image
        The aorta segmentation
    """
    output_map = torch_model_configs["total"]["output_map"]
    labels = ["aorta", "brachiocephalic_trunk", "subclavian_artery_left"]
    label_incides = [output_map[label] for label in labels]
    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(label_incides, start=1)
    }
    # Get the aorta segmentation
    aorta = sitk.GetArrayFromImage(total_seg)
    mask = np.isin(aorta, label_incides)
    aorta[~mask] = 0
    for label, new_label in label_map.items():
        aorta[aorta == label] = new_label
    mapped_indices = [int(x) for x in np.unique(aorta)]
    # Convert back to sitk image
    aorta_mapped = sitk.GetImageFromArray(aorta)
    aorta_mapped.CopyInformation(total_seg)
    # Remove erroneous segmentations
    aorta_mapped = _filter_largest_components(aorta_mapped, mapped_indices)
    return aorta_mapped


def _check_aorta_regions(vert_midlines: dict) -> tuple[bool, bool, bool]:
    """Check for the viability to measure the different aortic regions in the image
    Parameters
    ----------
    vert_midlines : dict
        The vertebral midlines
    Returns
    -------
    tuple[bool, bool, bool]
        The existince of the thoracic, abdominal, and descending aorta in the CT image
    """
    # T4 and at least T8 need to be in the image for thoracic to be measured
    thoracic = vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get(
        "vertebrae_T4_midline", False
    )  # L3 has to be in the image for abdominal to be measured
    abdominal = vert_midlines.get("vertebrae_L3_midline", False)
    # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
    descending = vert_midlines.get(
        "vertebrae_T12_midline", False
    ) and vert_midlines.get("vertebrae_T9_midline", False)
    return thoracic, abdominal, descending


def _find_aortic_region_endpoints(verts: list, vert_midlines: dict) -> tuple[int, int]:
    """Parse the midlines from the vert dict to get the start and endpoints of the section
    Parameters
    ----------
    verts : list
        The list of vertebrae to measure
    vert_midlines : dict
        The vertebral midlines
    Returns
    -------
    tuple[int, int]
        The start and end of the region to measure
    """
    midlines = [
        vert_midlines.get(f"vertebrae_{vert}_midline")
        for vert in verts
        if vert_midlines.get(f"vertebrae_{vert}_midline") is not None
    ]
    midlines = [m for m in midlines if m is not None]
    start = min(midlines)
    end = max(midlines)
    return start, end


def _measure_thoracic_diameters(
    thoracic_aorta_seg: np.ndarray, cpr: np.ndarray, centerline: np.ndarray
) -> dict[str, float]:
    """Specialty function to measure maximum diameter of the ascending aorta, aortic arch and descending aorta.
    :param thoracic_aorta_seg: the entire aortic segmentation
    :param cpr: the curved planar reformation of the aorta
    :param centerline: the centerline of the thoracic aorta
    """
    unique_cpr_labels = np.unique(cpr).tolist()
    # Need background(0), aorta(1), brachiocephalic trunk(2) and left subclavian artery(3) in cpr
    # in order to do the simple split
    all_needed_labels = [0, 1, 2, 3]
    split_using_cpr = unique_cpr_labels == all_needed_labels
    if split_using_cpr:
        asc_cpr, arch_cpr, desc_cpr = _split_thoracic_using_cpr(cpr, centerline)
        if asc_cpr is None:  # If splitting using the CPR fails, split using the seg
            asc_cpr, arch_cpr, desc_cpr = _split_thoracic_using_seg(
                thoracic_aorta_seg, cpr, centerline
            )
    else:
        asc_cpr, arch_cpr, desc_cpr = _split_thoracic_using_seg(
            thoracic_aorta_seg, cpr, centerline
        )
    diam_data = {}
    for cpr, prefix in zip(
        [asc_cpr, arch_cpr, desc_cpr], ["asc_aorta", "aortic_arch", "desc_aorta"]
    ):
        if cpr is not None:
            region_diams = measure_largest_cpr_diameter(
                cpr, AORTA_CROSS_SECTION_SPACING
            )
            region_diams.update(
                measure_mid_cpr_diameter(cpr, AORTA_CROSS_SECTION_SPACING)
            )
            diams = {f"{prefix}_{k}": v for k, v in region_diams.items()}
            prox_diam, _, _ = measure_cross_sectional_diameter(
                cpr[0], AORTA_CROSS_SECTION_SPACING, diff_threshold=5
            )
            diams[f"{prefix}_prox_diam"] = prox_diam
            diam_data.update(diams)
    return diam_data


def _split_thoracic_using_cpr(cpr: np.ndarray, centerline: np.ndarray) -> tuple:
    """Split the thoracic cpr into the ascending, arch, and descending regions using the brachiocephalic trunk
    and left subclavian artery
    :param cpr: the cpr numpy array
    :param centerline: the centerline used to create the cpr
    :return: the ascending, arch, and descending cprs
    """
    brach_start, subclavian_end, aorta_label = _define_aortic_arch_with_seg(cpr)
    if subclavian_end < brach_start:
        logger.warning(
            "Left Subclavian Artery found above Brachiocephalic Trunk. use centerline to split instead."
        )
        return None, None, None
    # Separate each cpr
    asc_cpr, asc_centerline = cpr[:brach_start].copy(), centerline[:brach_start].copy()
    asc_cpr = _remove_aortic_root(asc_cpr, asc_centerline, ROOT_LENGTH)
    arch_cpr = cpr[brach_start:subclavian_end].copy()
    desc_cpr = cpr[subclavian_end:].copy()
    # Set all cprs to be binary (0 = background, 1 = aorta)
    asc_cpr[asc_cpr != aorta_label] = 0
    arch_cpr[arch_cpr != aorta_label] = 0
    desc_cpr[desc_cpr != aorta_label] = 0
    return asc_cpr, arch_cpr, desc_cpr


def _define_aortic_arch_with_seg(cpr):
    aorta_label = 1
    brach_label = 2
    subclavian_label = 3
    brach_start = 0
    subclavian_end = 0
    cpr_length = len(cpr)
    # First find the first instance of the brachiocephalic trunk
    for slice_idx, cross_section in enumerate(cpr):
        if brach_label in cross_section:
            brach_start = slice_idx
            break
    # Now find the last instance of the subclavian artery
    for slice_idx, cross_section in enumerate(
        cpr[::-1]
    ):  # Go through in reverse so first appearance == last instance
        if subclavian_label in cross_section:
            subclavian_end = cpr_length - slice_idx
            break
    return brach_start, subclavian_end, aorta_label


def _split_thoracic_using_seg(
    thoracic_aorta_seg: np.ndarray, cpr: np.ndarray, centerline: np.ndarray
) -> tuple:
    """The original way we split the thoracic aorta into ascending, arch, and descending regions using the segmentation
    :param thoracic_aorta_seg: the entire aortic segmentation
    :param cpr: the cpr numpy array
    :param centerline: the centerline used to create the cpr
    """
    # Find the highest point of the aortic centerline
    arch_peak = round(float(np.argmin(centerline, axis=0)[0]))
    asc_cpr = cpr[:arch_peak]
    asc_centerline = centerline[:arch_peak]
    desc_cpr = cpr[arch_peak:]
    desc_centerline = centerline[arch_peak:]
    # find where the aorta splits into 2 distinct regions axially
    # This defines the minimum
    min_pixel_area = 100
    split = None
    for slice_idx, axial_image in enumerate(thoracic_aorta_seg):
        regions = _get_regions(axial_image)
        if len(regions) == 2:
            reg0 = regions[0]
            reg1 = regions[1]
            # If both sections of the aorta are sufficiently large,
            if reg0.area > min_pixel_area and reg1.area > min_pixel_area:
                split = slice_idx
                break
    if split is None:
        logger.debug("Aortic arch could not be defined. Thoracic measurement skipped.")
        return None, None, None
    # now remove the arch based on the split
    # First ascending
    asc = []
    new_asc_centerline = []
    max_asc_idx = 0
    for i in range(len(asc_cpr)):
        if asc_centerline[i][0] >= split:
            asc.append(asc_cpr[i])
            new_asc_centerline.append(asc_centerline[i])
            max_asc_idx = i
    if len(asc) > 0:
        asc = np.stack(asc, axis=0)
        new_asc_centerline = np.stack(new_asc_centerline, axis=0)
        asc_cpr = _remove_aortic_root(asc, new_asc_centerline, ROOT_LENGTH)
    else:
        asc_cpr = None
    # descending
    desc = []
    min_desc_idx = len(desc_cpr) - 1
    for i in range(len(desc_cpr)):
        if desc_centerline[i][0] >= split:
            desc.append(desc_cpr[i])
            if (
                i < min_desc_idx
            ):  # This should only happen once, the first time the centerline is below the split
                min_desc_idx = i
    if len(desc) > 0:
        desc_cpr = np.stack(desc, axis=0)
    else:
        desc_cpr = None
    if max_asc_idx < min_desc_idx:
        arch_cpr = cpr[max_asc_idx:min_desc_idx]
    else:
        arch_cpr = None
    return asc_cpr, arch_cpr, desc_cpr


def _remove_aortic_root(
    ascending_cpr: np.ndarray,
    ascending_centerline: np.ndarray,
    threshold: int,
    invert: bool = False,
) -> np.ndarray:
    """Remove the aortic root region from the ascending aorta cpr
    :param ascending_cpr: the cpr of the ascending aorta
    :param ascending_centerline: the centerline of the ascending aorta
    :param threshold: the distance in mm on the centerline to skip (aka. defined length of aortic root)
    :param invert: Return the aortic root region instead of the region without the aortic root
    """
    distances = np.sum((ascending_centerline - ascending_centerline[0]) ** 2, axis=1)
    aortic_root_cutoff = np.where(distances >= (threshold**2))[0][0]
    if invert:
        return ascending_cpr[:aortic_root_cutoff]
    else:
        return ascending_cpr[aortic_root_cutoff:]
