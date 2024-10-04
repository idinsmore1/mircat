from os import environ
from time import time
from functools import partial
from numpy import array_split
from loguru import logger
from tqdm import tqdm
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

from .nifti import NiftiMircato as NiftiMircato
from .nifti import TotalSegNotFoundError, BodySegNotFoundError, TissuesSegNotFoundError
from .aorta import calculate_aorta_stats as calculate_aorta_stats
from .total import calculate_total_segmentation_stats as calculate_total_stats
from .body_and_tissues import (
    calculate_body_and_tissues_stats as calculate_body_and_tissues_stats,
)
from .contrast_detection import predict_contrast as predict_contrast

from mircat_stats.configs import set_num_threads


def main(
    nifti_files: list[str],
    task_list: list[str],
    num_workers: int,
    num_threads: int,
    mark_complete: bool,
    gaussian: bool
):
    """Function that calculates statistics for a list of nifti files over a set number of workers
    Parameters
    ----------
    nifti_files : list[str]
        The list of nifti files to process
    task_list : list[str]
        The list of statistics tasks to calculate
    num_workers : int
        The number of workers to use
    num_threads : int
        The number of threads to use
    mark_complete : bool
        Whether to mark the statistics as complete
    gaussian : bool
        Whether to apply a gaussian smoothing to the image
    """
    logger.info(f"Processing statistics for {len(nifti_files)} nifti files")
    stats_start = time()
    with threadpool_limits(limits=num_threads):
        if num_workers > 1:
            pbar = tqdm(nifti_files, total=len(nifti_files), desc="Measuring Niftis", dynamic_ncols=True)
            nifti_stats = partial(single_nifti_stats, task_list=task_list, mark_complete=mark_complete, gaussian=gaussian)
            with Pool(num_workers) as pool:
                for _ in pool.imap_unordered(nifti_stats, nifti_files):
                    pbar.update(1)
        else:
            for nifti in tqdm(
                nifti_files,
                total=len(nifti_files),
                desc="Measuring Niftis",
                dynamic_ncols=True,
            ):
                single_nifti_stats(nifti, task_list, mark_complete, gaussian)
    stats_end = time()
    logger.info(
        "Statistics processing completed in {:.2f}s".format(stats_end - stats_start)
    )


def single_nifti_stats(
    input_nifti: str, task_list: list[str], mark_complete: bool, gaussian: bool
) -> None:
    """Calculate statistics for a single nifti file
    Parameters
    ----------
    nifti : Path
        The nifti file to calculate statistics for
    task_list : list[str]
        The list of statistics tasks to calculate
    mark_complete : bool
        Whether to mark the statistics as complete
    gaussian : bool
        Whether to apply a gaussian smoothing to the image
    """
    try:
        nifti = NiftiMircato(input_nifti)
        nifti.setup(task_list, gaussian)

        # Set up the default values
        header_data = nifti.header_data
        all_stats: dict = header_data
        contrast_time = 0
        total_time = 0
        aorta_time = 0
        tissues_time = 0
        total_stats = {}
        vert_midlines = {}
        all_completed = False
        # Run the statistics tasks
        if "contrast" in task_list:
            contrast_stats, contrast_time = predict_contrast(nifti)
            all_stats["contrast_completed"] = True
            all_stats.update(contrast_stats)

        if "total" in task_list:
            (total_stats, vert_midlines), total_time = calculate_total_stats(nifti)
            all_stats["total_completed"] = True
            all_stats.update(total_stats)
            all_stats.update(vert_midlines)

        if "aorta" in task_list:
            if "total" in task_list and total_stats and vert_midlines:
                if (
                    total_stats.get("aorta_volume_cm3") is not None
                    and all_stats.get("ct_direction", "AX") == "AX"
                ):
                    aorta_stats, aorta_time = calculate_aorta_stats(nifti, vert_midlines)
                    all_stats["aorta_completed"] = True
                    all_stats.update(aorta_stats)
                else:
                    logger.warning(
                        "Either aorta does not exist or the CT direction is not AX, so aorta will be skipped."
                    )
            elif "aorta" in task_list and "total" not in task_list:
                if nifti.stats_exist and nifti.vert_midlines != {}:
                    aorta_stats, aorta_time = calculate_aorta_stats(
                        nifti, nifti.vert_midlines
                    )
                    all_stats["aorta_completed"] = True
                    all_stats.update(aorta_stats)
                else:
                    logger.warning(
                        "Vertebrae midlines do not exist, so total stats will be ran."
                    )
                    (total_stats, vert_midlines), total_time = calculate_total_stats(nifti)
                    all_stats["total_completed"] = True
                    all_stats.update(total_stats)
                    all_stats.update(vert_midlines)
                    # Now run aorta stats
                    if (
                        total_stats.get("aorta_volume_cm3") is not None
                        and all_stats.get("ct_direction", "AX") == "AX"
                    ):
                        aorta_stats, aorta_time = calculate_aorta_stats(
                            nifti, vert_midlines
                        )
                        all_stats["aorta_completed"] = True
                        all_stats.update(aorta_stats)
                    else:
                        logger.warning(
                            "Either aorta does not exist or the CT direction is not AX, so aorta will be skipped."
                        )

        if "tissues" in task_list:
            if vert_midlines:
                tissues_data, tissues_time = calculate_body_and_tissues_stats(
                    nifti, vert_midlines
                )
                all_stats["tissues_completed"] = True
                all_stats.update(tissues_data)
            elif nifti.stats_exist and nifti.vert_midlines != {}:
                tissues_data, tissues_time = calculate_body_and_tissues_stats(
                    nifti, nifti.vert_midlines
                )
                all_stats["tissues_completed"] = True
                all_stats.update(tissues_data)
            else:
                logger.warning(
                    "Vertebrae midlines do not exist, so total stats will be ran."
                )
                (total_stats, vert_midlines), total_time = calculate_total_stats(nifti)
                all_stats["total_completed"] = True
                all_stats.update(total_stats)
                all_stats.update(vert_midlines)
                # Now run tissues stats
                tissues_data, tissues_time = calculate_body_and_tissues_stats(
                    nifti, vert_midlines
                )
                all_stats["tissues_completed"] = True
                all_stats.update(tissues_data)

        if mark_complete:
            all_completed = True
            all_stats["total_completed"] = True
            all_stats["contrast_completed"] = True
            all_stats["aorta_completed"] = True
            all_stats["tissues_completed"] = True
        elif set(task_list) == set(["total", "contrast", "aorta", "tissues"]):
            all_completed = True
        elif nifti.stats_exist:
            all_stats.update(nifti.stats)
            if (
                all_stats.get("total_completed")
                and all_stats.get("contrast_completed")
                and all_stats.get("aorta_completed")
                and all_stats.get("tissues_completed")
            ):
                all_completed = True
        nifti.write_stats_to_file(all_stats, all_completed)
        logger.success(
            f"Stats for {nifti}, complete: {all_completed}. \n\tContrast Pred Time: {contrast_time}s. Total Stats Time: {total_time}s. Aorta Stats Time: {aorta_time}s. Tissue Stats time: {tissues_time}s.\n\tStats file -> {nifti.output_file}\n",
            extra={
                "key": "statistics",
                "input_nifti": str(nifti),
                "completed": True,
                "failed_reason": None,
                "output_file": str(nifti.output_file),
                "contrast_pred_time": contrast_time,
                "total_stats_time": total_time,
                "aorta_stats_time": aorta_time,
                "body_and_tissues_time": tissues_time,
            },
        )
    except TotalSegNotFoundError:
        logger.error(
            f"Stats for {input_nifti} failed. Total segmentation not found.",
            extra={
                "key": "statistics",
                "input_nifti": input_nifti,
                "completed": False,
                "failed_reason": "no_total",
                "output_file": None,
                "contrast_pred_time": None,
                "total_stats_time": None,
                "aorta_stats_time": None,
                "body_and_tissues_time": None,
            },
        )
    except BodySegNotFoundError:
        logger.error(
            f"Stats for {input_nifti} failed. Body segmentation not found.",
            extra={
                "key": "statistics",
                "input_nifti": input_nifti,
                "completed": False,
                "failed_reason": "no_body",
                "output_file": None,
                "contrast_pred_time": None,
                "total_stats_time": None,
                "aorta_stats_time": None,
                "body_and_tissues_time": None,
            },
        )
    except TissuesSegNotFoundError:
        logger.error(
            f"Stats for {input_nifti} failed. Tissues segmentation not found.",
            extra={
                "key": "statistics",
                "input_nifti": input_nifti,
                "completed": False,
                "failed_reason": "no_tissues",
                "output_file": None,
                "contrast_pred_time": None,
                "total_stats_time": None,
                "aorta_stats_time": None,
                "body_and_tissues_time": None,
            },
        )
    except Exception as e:
        logger.error(
            f"Stats for {input_nifti} failed. {e}",
            extra={
                "key": "statistics",
                "input_nifti": input_nifti,
                "completed": False,
                "failed_reason": "unknown",
                "output_file": None,
                "contrast_pred_time": None,
                "total_stats_time": None,
                "aorta_stats_time": None,
                "body_and_tissues_time": None,
            },
        )
