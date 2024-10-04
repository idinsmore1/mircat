import json
import numpy as np
import dicom2nifti
import dicom2nifti.settings as settings
from warnings import filterwarnings

from datetime import datetime
from dicom2nifti.convert_dir import _remove_accents
from loguru import logger
from operator import itemgetter
from pydicom import dcmread
from pathlib import Path
from shutil import copyfile, SameFileError

from mircat_stats.configs.logging import timer


# Set up the constant values that we will need for the dicom header data
ORIENTATIONS = {
    "COR": [1, 0, 0, 0, 0, -1],
    "SAG": [0, 1, 0, 0, 0, -1],
    "AX": [1, 0, 0, 0, 1, 0],
}
MIN_FILE_COUNT = 40
# Dictionary of tag names and tag keys, and if the tag is a sequence
IMPORTANT_ATTRIBUTES = {
    "PatientID": "mrn",
    "AccessionNumber": "accession",
    "SeriesDescription": "series_name",
    "SeriesNumber": "series_number",
    "ImageOrientationPatient": "ct_direction",
    "ImageType": "image_type",
    "PatientSex": "sex",
    "PatientSize": "height_at_scan_m",
    "PatientWeight": "weight_at_scan_kg",
    "PregnancyStatus": "pregnancy_status",
    "PatientAge": "age_at_scan",
    "PatientBirthDate": "birthday",
    "AcquisitionDate": "scan_date",
    "PixelSpacing": None,
    "SliceThickness": "slice_thickness_mm",
    "Rows": "num_rows",
    "Columns": "num_columns",
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "manufacturer_model",
    "KVP": "kvp",
    "SequenceName": "sequence_name",
    "ProtocolName": "protocol_name",
    "ContrastBolusAgent": "contrast_bolus_agent",
    "ContrastBolusRoute": "contrast_bolus_route",
    "MultienergyCTAcquisitionSequence": "multienergy_ct",
    "ProcedureCodeSequence": None,
    "StudyInstanceUID": "study_uid",
    "SeriesInstanceUID": "series_uid",
}


# Set the dicom to nifti settings
settings.disable_validate_slice_increment()
settings.enable_resampling = True
settings.resampling_order = 1
settings.reorient_nifti = True


class NotADicomFolderError(Exception):
    "Raised when the folder is not a valid dicom folder"

    pass


class InsufficientDicomFilesError(Exception):
    "Raised when there are not enough dicom files in a folder"

    pass


class IncompleteDicomReferenceError(Exception):
    "Raised when the dicom information is incomplete"

    pass


class DicomFolder:
    "Class representing a Dicom Folder to be converted and store a reference"

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def __str__(self):
        return str(self.path)

    def _find_dicoms_in_folder(self) -> list[Path]:
        dicom_files = list(self.path.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(self.path.glob("*"))
        return dicom_files

    def _check_folder(self) -> None:
        "Check the contents of the folder to ensure it is a valid dicom folder"
        min_file_count = MIN_FILE_COUNT
        if not self.path.is_dir():
            raise NotADicomFolderError
        dicom_files = self._find_dicoms_in_folder()
        if len(dicom_files) < min_file_count:
            raise InsufficientDicomFilesError
        self.reference_dicom = dicom_files[0]
        self.reference_dicom_name = dicom_files[0].parts[-1]
        self.series_numbers = set(
            [getattr(dcmread(dicom), "SeriesNumber") for dicom in dicom_files]
        )
        self.series_name = _remove_accents(
            getattr(dcmread(self.reference_dicom), "SeriesDescription", "")
        )

    def _generate_reference_dict(self) -> None:
        "Load reference dicom header and generate a dictionary containing critical data"
        ref_header = dcmread(self.reference_dicom)
        naming_keys = ["mrn", "accession", "series_name"]
        ref_data = {}
        important_attributes = IMPORTANT_ATTRIBUTES
        orientations = ORIENTATIONS
        for tag, column in important_attributes.items():
            value = getattr(ref_header, tag, None)
            if value == "":
                value = None
            if value is not None:
                match tag:
                    case "SeriesDescription":
                        value = _remove_accents(value)
                    case "PatientID":
                        value = str(value).zfill(8)
                    case "ImageOrientationPatient":
                        value = None
                        orientation = np.round(getattr(ref_header, tag))
                        for k, direction in orientations.items():
                            if np.array_equal(orientation, direction):
                                value = k
                                break
                    case "ImageType":
                        value = "_".join(value)
                    case tag if tag in ["PatientBirthDate", "AcquisitionDate"]:
                        value = datetime.strptime(value, "%Y%m%d").date()
                        value = value.strftime("%Y-%m-%d")
                    case "PatientAge":
                        value = int(value[:-1])
                    case "PixelSpacing":
                        length, width = value
                        ref_data["length_mm"] = length
                        ref_data["width_mm"] = width
                        continue
                    case "PatientSex":
                        if value == "M":
                            value = 0
                        elif value == "F":
                            value = 1
                        else:
                            value = None
                    case tag if tag in ["ContrastBolusAgent", "ContrastBolusRoute"]:
                        value = 1
                    case "MultienergyCTAcquisitionSequence":
                        value = 1
                    case "PregnancyStatus":
                        value = int(value)
                        if value == 4:
                            value = None
                        elif value == 1:
                            value = 0
                        elif value == 2:
                            value = 1
                        elif value == 3:
                            value = 2
                    case tag if tag in ["SeriesNumber", "Rows", "Columns"]:
                        value = int(value)
                    case tag if tag in ["PatientSize", "PatientWeight"]:
                        value = float(value)
                    case tag if tag in ["SequenceName", "ProtocolName"]:
                        value = str(value).lower()
                    case "ProcedureCodeSequence":
                        seq = value[0]
                        ref_data["procedure_code"] = seq.CodeValue
                        ref_data["procedure_desc"] = seq.CodeMeaning
                        continue
                    case _:
                        value = value
                ref_data[column] = value
            elif tag == "ProcedureCodeSequence":
                ref_data["procedure_code"] = None
                ref_data["procedure_desc"] = None
            elif tag == "PixelSpacing":
                ref_data["length_mm"] = None
                ref_data["width_mm"] = None
            else:
                ref_data[column] = None
        if ref_data["slice_thickness_mm"] is None:
            ref_data["slice_thickness_mm"] = 5
        # Perform two quick checks for data existence
        ref_data["is_mip"] = ("mip" in ref_data.get("series_name", "").lower()) or (
            ref_data.get("slice_thickness_mm", 5) > 5
        )
        ref_data["is_airc"] = "ai-rad" in ref_data.get("series_name", "").lower()
        self.ref_data = ref_data
        if all([x is None for x in ref_data.values()]) or any(
            [ref_data.get(key) is None for key in naming_keys]
        ):
            raise IncompleteDicomReferenceError

    @timer
    def _convert(self, nifti_dir: Path) -> None:
        """Convert the dicoms to nifti after checks have been passed
        Parameters
        ----------
        nifti_dir : Path
            The directory where the nifti file will be saved
        """
        if not nifti_dir.exists():
            nifti_dir.mkdir(parents=True)

        try:
            copyfile(self.reference_dicom, nifti_dir / self.reference_dicom_name)
        except SameFileError:  # This sometimes gets raised, not sure why
            pass
        # Write the reference dict to a json file called header_info.json
        with (nifti_dir / "header_info.json").open("w") as f:
            json.dump(self.ref_data, f, indent=4)
        # Convert the dicoms to nifti
        dicom2nifti.convert_directory(
            self.path, nifti_dir, compression=True, reorient=True
        )

    def convert_to_nifti(self, output_dir: Path, only_ax: bool, no_mip: bool) -> None:
        """Converts the dicom folder to a nifti file.
        Parameters
        ----------
        output_dir : Path
            The base directory where the nifti file will be saved
        only_ax : bool
            If True, only will be saved if the series is an axial view
        no_mip : bool
            If True, MIP series will not be saved
        """
        extra_dict = {
            "key": "dicom_to_nifti",
            "input_dir": str(self.path),
            "converted": False,
            "output_dir": None,
            "output_files": None,
            "failed_reason": None,
        }
        try:
            self._check_folder()
            self._generate_reference_dict()
            if only_ax and self.ref_data["ct_direction"] != "AX":
                extra_dict["failed_reason"] = "not_axial"
                logger.error(
                    f"{self.path} is not an axial scan and only_ax = True, will not convert",
                    extra=extra_dict,
                )
                return
            if no_mip and self.ref_data["is_mip"]:
                extra_dict["failed_reason"] = "likely_mip"
                logger.error(
                    f"{self.path} is likely a MIP scan and no_mip = True, will not convert",
                    extra=extra_dict,
                )
                return
            if self.ref_data["is_airc"]:
                extra_dict["failed_reason"] = "ai_rad"
                logger.error(
                    f"{self.path} is an AI-RAD scan and will not convert",
                    extra=extra_dict,
                )
                return
            mrn, accession, series_name = itemgetter("mrn", "accession", "series_name")(
                self.ref_data
            )
            first_two_mrn_digits = mrn[:2]
            nifti_dir = (
                output_dir / first_two_mrn_digits / mrn / accession / series_name
            )
            _, convert_time = self._convert(nifti_dir)
            nifti_files = [
                f"{n}_{self.series_name}.nii.gz" for n in self.series_numbers
            ]
            # Update the extra_dict with the output directory and files
            extra_dict["converted"] = True
            extra_dict["output_dir"] = str(nifti_dir)
            extra_dict["output_files"] = nifti_files
            logger.success(
                f"{self.path} converted to nifti in {convert_time:.2f} seconds. Saved to {nifti_dir}",
                extra=extra_dict,
            )
        except NotADicomFolderError:
            extra_dict["failed_reason"] = "not_a_directory"
            logger.error(f"{self.path} is not a directory", extra=extra_dict)
        except InsufficientDicomFilesError:
            extra_dict["failed_reason"] = "too_few_files"
            logger.error(
                f"Less than {MIN_FILE_COUNT} DICOM files found in {self.path}",
                extra=extra_dict,
            )
        except IncompleteDicomReferenceError:
            extra_dict["failed_reason"] = "incomplete_reference"
            logger.error(
                f"Dicom reference data is incomplete for {self.path}", extra=extra_dict
            )

        except SameFileError as e:
            extra_dict["failed_reason"] = "same_file_error"
            logger.warning(
                f"Nifti file already exists for {self.path}, will not convert. Error: {str(e)}",
                extra=extra_dict,
            )
        except dicom2nifti.exceptions.ConversionValidationError:
            extra_dict["failed_reason"] = "conversion_validation_error"
            logger.error(
                f"Dicom conversion could not be validated for {self.path}",
                extra=extra_dict,
            )
        except PermissionError as e:
            extra_dict["failed_reason"] = "permission_error"
            logger.error(
                f"Dicom folder {self.path} could not be accessed due to a permission error {str(e)}",
                extra=extra_dict,
            )
        except AttributeError as e:
            extra_dict["failed_reason"] = "attribute_error"
            logger.error(
                f"Dicom folder {self.path} does not have the necessary attributes",
                extra=extra_dict,
            )
        except Exception as e:
            extra_dict["failed_reason"] = type(e).__name__
            logger.error(
                f"A {type(e).__name__} error occured while converted the dicom folder {str(e)}",
                extra=extra_dict,
            )
