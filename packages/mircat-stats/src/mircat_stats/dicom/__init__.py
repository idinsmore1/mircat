import json
from time import time
from loguru import logger
from functools import partial
from pathlib import Path
from pydicom.errors import InvalidDicomError
from multiprocessing import Pool
from tqdm import tqdm
from .dicom_folder import DicomFolder as DicomFolder
from mircat_stats.configs.statistics import stats_output_keys


def convert_dicom_folders_to_nifti(
    dicom_folders: list,
    output_dir: Path,
    num_workers: int,
    only_ax: bool,
    no_mip: bool,
    verbose: bool,
):
    """Main function to be passed to mircato CLI to convert DICOM folders to NIfTI
    Parameters
    ----------
    dicom_folders : list
        List of paths to DICOM folders
    output_dir : Path
        Path to the output directory
    num_workers : int
        Number of workers to use for the conversion
    only_ax : bool
        If True, only axial slices will be converted
    verbose : bool
        If True, verbose output will be printed
    """
    converter = partial(
        _process_dicom_folder, output_dir=output_dir, only_ax=only_ax, no_mip=no_mip
    )
    logger.info(
        f"Converting {len(dicom_folders)} dicom folders to nifti with {num_workers} workers"
    )
    start_time = time()
    with Pool(num_workers) as pool:
        if verbose:
            dicom_iterator = pool.imap_unordered(converter, dicom_folders)
        else:
            dicom_iterator = tqdm(
                pool.imap_unordered(converter, dicom_folders),
                total=len(dicom_folders),
                dynamic_ncols=True,
                desc="Converting DICOM folders to NIfTI",
            )
        for _ in dicom_iterator:
            pass
    end_time = time()
    logger.info(
        f"Conversion of {len(dicom_folders)} dicom folders to nifti completed in {end_time - start_time:.2f}s"
    )


def _process_dicom_folder(
    dicom_folder: str, output_dir: Path, only_ax: bool, no_mip: bool
) -> None:
    """Helper function to process a single dicom folder
    Parameters
    ----------
    dicom_folder : str
        Path to the dicom folder
    output_dir : Path
        Path to the output directory
    only_ax : bool
        If True, only axial slices will be converted
    no_mip : bool
        If True, mip series will not be converted
    """
    dicom_folder = DicomFolder(dicom_folder)
    dicom_folder.convert_to_nifti(output_dir, only_ax, no_mip)



def update(niftis: list, num_workers: int):
    """Update the header and stats data for a NIfTI file to the latest version
    Parameters
    ----------
    niftis : list
        List of paths to NIfTI files
    num_workers : int
        Number of workers to use for the update
    """
    niftis = [Path(nifti) for nifti in niftis]
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(update_header_and_stats, niftis),
                total=len(niftis),
                desc="Updating NIfTI headers and stats",
                dynamic_ncols=True,
            ):
                pass
    else:
        for nifti in tqdm(
            niftis,
            total=len(niftis),
            desc="Updating NIfTI headers and stats",
            dynamic_ncols=True,
        ):
            update_header_and_stats(nifti)


def update_header_and_stats(nifti: Path):
    """Function to update the header and calculate statistics for a nifti file
    Parameters
    ----------
    nifti : Path
        Path to the nifti file
    """
    # Establish all the paths that need to be known
    nifti_folder = nifti.parent
    header_file = nifti_folder / "header_info.json"
    nifti_name = nifti.name.partition(".")[0]
    seg_folder = nifti_folder.absolute() / f"{nifti_name}_segs"
    output_file = seg_folder / f"{nifti_name}_stats.json"

    nifti_as_dicom = DicomFolder(nifti_folder)
    dicom_files = nifti_as_dicom._find_dicoms_in_folder()
    # Some nifti folders will have a dicom with .dcm extension, others will not
    if len(dicom_files) == 1:  # This will always be the case if .dcm extension is present
        nifti_as_dicom.reference_dicom = dicom_files[0]
        nifti_as_dicom._generate_reference_dict()
    else:
        for dicom_file in dicom_files:
            try:
                nifti_as_dicom.reference_dicom = dicom_file
                nifti_as_dicom._generate_reference_dict()
                break
            except InvalidDicomError:
                continue
    if not hasattr(nifti_as_dicom, "ref_data"):
        logger.error(
            f"Could not find a valid reference dicom file for {nifti}",
            extra={
                "key": "update",
                "input_nifti": str(nifti),
                "output_file": str(output_file),
                "completed": False,
                "failed_reason": "No valid reference dicom found",
            }
        )
        return
    header_data = nifti_as_dicom.ref_data
    # Update the header file
    if header_file.exists():
        with header_file.open() as f:
            old_header = json.load(f)
        header_data.update(old_header)
    with header_file.open("w") as f:
        json.dump(header_data, f, indent=4)
    # If the stats file exists, update it
    if output_file.exists():
        with output_file.open() as f:
            stats = json.load(f)
        stats.update(header_data)
        stats = {k: stats.get(k) for k in stats_output_keys}
        with output_file.open("w") as f:
            json.dump(stats, f, indent=4)
    logger.success(
        f"Header and stats updated for {nifti}",
        extra={
            "key": "update",
            "input_nifti": str(nifti),
            "output_file": str(output_file),
            "completed": True,
            "failed_reason": None,
        },
    )
