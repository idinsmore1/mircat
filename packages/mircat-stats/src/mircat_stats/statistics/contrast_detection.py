"""Contrast detection based on the Comp2Comp model with our segmentations"""

import os
import pickle
import numpy as np
import scipy
import SimpleITK as sitk
from scipy import ndimage as ndi
from xgboost import XGBClassifier

from loguru import logger
from warnings import filterwarnings, catch_warnings
from mircat_stats.statistics.nifti import NiftiMircato
from mircat_stats.configs.models import torch_model_configs
from mircat_stats.configs.logging import get_project_root, timer

# filterwarnings("ignore")
CONTRAST_PREDICTION_DICT = {0: "non_contrast", 1: "arterial", 2: "venous", 3: "delayed"}
LABEL_NAMES = [
    "aorta",
    "inferior_vena_cava",
    "portal_and_splenic_vein",
    "heart",
    "kidney_left",
    "kidney_right",
    "adrenal_gland_right",
]
FEATURE_LIST = [
    "aorta",
    "inferior_vena_cava",
    "portal_and_splenic_vein",
    "kidney_left",
    "kidney_right",
    "kidney_left_hull",
    "kidney_right_hull",
    "aorta_portal_vein",
    "aorta_ivc",
]


@timer
def predict_contrast(nifti: NiftiMircato) -> dict[str : str | str : float]:
    """Predict contrast phase of an image using radiomics data
    Parameters
    ----------
    nifti : NiftiMircato
        The nifti file to predict contrast for
    Returns
    -------
    dict
        The predicted contrast phase and the probability
    """
    try:
        # Load in the images and convert the segmentation to an array
        img = nifti.original_ct
        seg = nifti.total_seg
        seg_arr = sitk.GetArrayFromImage(nifti.total_seg)
        # Set the label map

        label_map = torch_model_configs["total"]["output_map"]
        binary_data = {}
        # Create all the binary masks
        label_names = LABEL_NAMES
        for name in label_names:
            label = label_map[name]
            binary_arr = _binarize_array(seg_arr, label)
            if name == "portal_and_splenic_vein":
                struct = np.ones((1, 1, 1))
            else:
                struct = np.ones((3, 3, 3))
            eroded_arr = ndi.binary_erosion(binary_arr, structure=struct).astype(
                np.uint8
            )
            if name == "portal_and_splenic_vein":
                if eroded_arr.sum() < 500:
                    eroded_arr = binary_arr
            eroded_sitk = sitk.GetImageFromArray(eroded_arr)
            eroded_sitk.CopyInformation(seg)
            binary_data[name] = eroded_sitk
            if name == "kidney_left" or name == "kidney_right":
                if eroded_arr.sum() > 0:
                    try:
                        mask = eroded_arr.copy()
                        hull = _fill_hull(mask)
                        hull = hull * (mask == 0)
                        struct = np.ones((3, 3, 3))
                        hull = ndi.binary_erosion(hull, structure=struct).astype(
                            np.uint8
                        )
                        hull_sitk = sitk.GetImageFromArray(hull)
                        hull_sitk.CopyInformation(seg)
                        binary_data[f"{name}_hull"] = hull_sitk
                    except Exception as e:
                        logger.error(e)
                        hull = np.zeros_like(eroded_arr)
                        hull_sitk = sitk.GetImageFromArray(hull)
                        hull_sitk.CopyInformation(seg)
                        binary_data[f"{name}_hull"] = hull_sitk
                else:
                    hull = np.zeros_like(eroded_arr)
                    hull_sitk = sitk.GetImageFromArray(hull)
                    hull_sitk.CopyInformation(seg)
                    binary_data[f"{name}_hull"] = hull_sitk
        # Get the statistics for each label
        statistics = {
            label: _get_sitk_stats(binary_data[label], img) for label in binary_data
        }
        statistics["aorta_portal_vein"] = (
            statistics["aorta"][:3] - statistics["portal_and_splenic_vein"][:3]
        )
        statistics["aorta_ivc"] = (
            statistics["aorta"][:3] - statistics["inferior_vena_cava"][:3]
        )
        features = []
        feature_list = FEATURE_LIST
        contrast_prediction_dict = CONTRAST_PREDICTION_DICT
        for feature in feature_list:
            features.extend(list(statistics[feature]))
        model = _load_model()
        try:
            y_pred_proba = model.predict_proba([features])[0]
            y_pred = np.argmax(y_pred_proba)
            pred_val = contrast_prediction_dict.get(y_pred)
            pred_prob = round(float(y_pred_proba[y_pred]), 3)
            return {"contrast_pred": pred_val, "contrast_prob": pred_prob}
        except Exception as e:
            logger.error(f"Contrast prediction failed: {e}")
            return {"contrast_pred": None, "contrast_prob": None}
    except Exception as e:
        logger.error(f"Contrast prediction failed: {e}")
        return {"contrast_pred": None, "contrast_prob": None}


def _load_model():
    model_path = os.path.join(get_project_root(), "models/xgboost.json")
    # with catch_warnings():
    #     filterwarnings("ignore")
    # with open(model_path, "rb") as f:
    #     model = pickle.load(f)
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def _binarize_array(arr: np.ndarray, label: int) -> np.ndarray:
    """Binarize a numpy array using a specified label"""
    return (arr == label).astype(np.uint8)


def _fill_hull(image: np.ndarray) -> np.ndarray:
    "Create the convex hull for a kidney segmentation"
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img


def _get_sitk_stats(seg: sitk.Image, img: sitk.Image):
    stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    stats_filter.Execute(seg, img)
    if 1 not in stats_filter.GetLabels():
        return np.repeat(np.nan, 6)
    max_val = stats_filter.GetMaximum(1)
    min_val = stats_filter.GetMinimum(1)
    mean_val = stats_filter.GetMean(1)
    median_val = stats_filter.GetMedian(1)
    std_val = stats_filter.GetStandardDeviation(1)
    variance_val = stats_filter.GetVariance(1)
    return np.array([max_val, min_val, mean_val, median_val, std_val, variance_val])
