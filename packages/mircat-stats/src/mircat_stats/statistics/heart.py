from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.segmentation import Segmentation


def calculate_heart_stats(nifti: MircatNifti) -> dict[str, float]:
    """
    Calculate the heart statistics for the given NIfTI file.

    :param nifti: The MircatNifti object
    :return: The dictionary of heart statistics
    """
    

class Heart(Segmentation):
    """
    The heart segmentation class
    """
    def __init__(self, nifti: MircatNifti):
        """
        Initialize the heart segmentation class.

        :param nifti: The MircatNifti object
        """
        super().__init__(nifti, ["heart"])
        self.extract_segmentation_bounding_box(5)