from mircat_stats.configs.models import torch_model_configs
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.nifti import NiftiMircato
from mircat_stats.statistics.utils import _calculate_3d_volumes, _calculate_2d_areas


@timer
def calculate_body_and_tissues_stats(
    nifti: NiftiMircato, vert_midlines: dict
) -> dict[str:float]:
    """Calculate the body and tissue statistics
    Parameters
    ----------
    nifti : NiftiMircato
        The nifti file to calculate statistics for
    vert_midlines : dict
        The vertebral midlines
    Returns
    -------
    dict
        The statistics for the body and tissues
    """
    body_map = torch_model_configs["body"]["output_map"]
    tissues_map = torch_model_configs["tissues"]["output_map"]
    # Set the segmentations
    body = nifti.body_seg
    tissues = nifti.tissues_seg
    # Get the midlines for the vertebrae of interest
    verts_of_interest = ["L1", "L3", "L5"]
    vert_indices = {
        v: vert_midlines.get(f"vertebrae_{v}_midline")
        for v in verts_of_interest
        if vert_midlines.get(f"vertebrae_{v}_midline") is not None
    }
    # Get the stats for the relevant regions
    output_stats = {}
    is_abdominal = "L1" in vert_indices and "L5" in vert_indices
    output_stats["abdominal_scan"] = int(is_abdominal)
    # Get the volumes for the entire scan
    prefix = "total_"
    body_volumes = _calculate_3d_volumes(body, body_map, prefix)
    tissues_volumes = _calculate_3d_volumes(tissues, tissues_map, prefix)
    output_stats.update(body_volumes)
    output_stats.update(tissues_volumes)
    # get the abdominal region values
    if is_abdominal:
        prefix = "abdominal_"
        endpoints = (
            vert_midlines.get("vertebrae_L5_midline"),
            vert_midlines.get("vertebrae_L1_midline") + 1,
        )
        abd_body_volumes = _calculate_3d_volumes(body, body_map, prefix, endpoints)
        abd_tissues_volumes = _calculate_3d_volumes(
            tissues, tissues_map, prefix, endpoints
        )
        output_stats.update(abd_body_volumes)
        output_stats.update(abd_tissues_volumes)
    # Get the vertebrae specific values
    for vert, midline in vert_indices.items():
        prefix = f"{vert}_"
        body_vert_stats = _calculate_2d_areas(
            body, body_map, midline, prefix, get_perimeter=True
        )
        tissues_vert_stats = _calculate_2d_areas(
            tissues, tissues_map, midline, prefix, get_perimeter=False
        )
        output_stats.update(body_vert_stats)
        output_stats.update(tissues_vert_stats)
    return output_stats
