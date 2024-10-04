import os
import time
from gc import collect
from itertools import islice
from math import ceil
from traceback import format_exc
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import autocast

from loguru import logger
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    MetaTensor,
    decollate_batch,
    set_track_meta,
)
from monai.data.utils import list_data_collate
from monai.inferers import SlidingWindowInfererAdapt
from monai.networks.nets import UNet
from monai import transforms as mt

from mircat_stats.configs.logging import get_project_root
from mircat_stats.configs.models import torch_model_configs
# from mircat_stats.configs import set_num_threads


MODELS_DIR = Path(__file__).parent / "models"


def segment_niftis(
    nifti_list: list,
    task_list: list,
    device: str,
    cache_num: int,
    sw_batch_size: int,
    threads: int,
    num_dataloader_workers: int,
) -> None:
    """Run inference on the given list of NIfTI files for a set of segmentation tasks.
    :param nifti_list: the list of NIfTI files to segment
    :param task_list: the list of tasks to run ['total', 'tissues', 'body']
    :param device: string defining the device to run model on
    :param cache_num: the number of NIfTI files to cache at one time for inference
    :param sw_batch_size: the sliding window batch size during inference
    :param threads: the number of threads for multi-threaded ops
    :param num_dataloader_workers: the number of worker processes to use loading images
    :return: None
    """
    # Set segmentation environment
    set_track_meta(True)
    # set_num_threads(threads)
    # Validate the nifti files - write logic later
    # validated_niftis = validate_niftis(nifti_list)
    validated_niftis = [{"image": nifti} for nifti in nifti_list]
    nifti_iter = iter(validated_niftis)
    torch_device = torch.device(device)
    logger.info(
        f'Running segmentation on {len(validated_niftis)} Nifti file{"s" if len(validated_niftis) > 1 else ""}'
    )
    batch_num = 1
    total_batches = ceil(len(validated_niftis) / cache_num)
    while True:
        batch = list(islice(nifti_iter, cache_num))
        if not batch:
            break
        for task in task_list:
            logger.info(
                f"Running {task} segmentation on batch {batch_num} of {total_batches}"
            )
            task_specific_segmentation(
                batch, task, torch_device, sw_batch_size, num_dataloader_workers
            )
            collect()
        batch_num += 1


def task_specific_segmentation(
    nifti_data: list[dict],
    task: str,
    device: torch.device,
    sw_batch_size: int,
    num_dataloader_workers: int,
) -> None:
    """Runs inference on a list of nifti files for a specific task
    :param nifti_data: a list of dictionaries in monai format containing nifti paths
    :param task: the specific inference task ['total', 'tissues', 'body']
    :param device: the device to run the inference on
    :param sw_batch_size: sliding window batch size
    :param num_dataloader_workers: the number of dataloader workers
    :return: None
    """
    # Get the task-specific configurations
    task_config = torch_model_configs[task]
    preprocessing, cpu_postprocessing, default_postprocessing = _create_transforms(
        task_config, device
    )
    dataloader = _create_dataloader(nifti_data, preprocessing, num_dataloader_workers)
    model = _load_model(task, device)
    # SlidingWindowInfererAdapt will automatically switch to CPU if cuda is out of memory
    inferer = SlidingWindowInfererAdapt(
        roi_size=task_config["patch_size"], sw_batch_size=sw_batch_size, overlap=0.5
    )
    # Run the inference using task configurations
    if device == torch.device('cpu'):
        autocast_device = 'cpu'
    else:
        autocast_device = 'cuda'
    with autocast(device_type=autocast_device):  # All models used AMP for training
        with torch.no_grad():
            for data in dataloader:
                image_meta = data["image"].meta
                if image_meta.get("skip", False):
                    continue
                images = data["image"].to(device)
                # Run inference
                start_time = time.time()
                pred = inferer(inputs=images, network=model)
                inference_time = round(time.time() - start_time, 2)
                data["pred"] = pred
                del images  # delete the original images to save memory
                processing_start = time.time()
                # If the prediction is on CPU (== -1), we just go right to CPU postprocessing
                if data["pred"].get_device() != -1:
                    try:
                        out = [
                            default_postprocessing(i) for i in decollate_batch(data)
                        ][0]
                    except RuntimeError:
                        print("switching postprocessing to CPU")
                        preprocessing = _reset_preprocessing(preprocessing)
                        out = [cpu_postprocessing(i) for i in decollate_batch(data)][0]
                else:
                    out = [cpu_postprocessing(i) for i in decollate_batch(data)][0]
                output_file = _write_segmentation_to_file(out, image_meta, task)
                processing_time = round(time.time() - processing_start, 2)
                del pred
                del out
                if output_file is not None:
                    logger.success(
                        f"""{image_meta["filename_or_obj"][0]}: {task.upper()}\n\tInference: {inference_time}s, Processing: {processing_time}s\n\tOutput: {output_file}""",
                        extra={
                            "key": "segmentation",
                            "input_nifti": image_meta["filename_or_obj"][0],
                            "task": task,
                            "completed": True,
                            "failed_reason": None,
                            "output_nifti": output_file,
                            "inference_time": inference_time,
                            "processing_time": processing_time,
                        },
                    )


def _create_transforms(
    task_config: dict, device: torch.device
) -> tuple[mt.Compose, mt.Compose, mt.Compose]:
    """Creates transforms for a specific task
    :param task_config: dictionary containing the configuration of the task
    :param device: the device to run postprocessing on (cpu or cuda)
    :return: a tuple containing the preprocessing, cpu postprocessing, and default postprocessing transforms
    """
    if task_config["name"] == "total":
        # Define the preprocessing pipeline
        preprocessing = mt.Compose(
            [
                mt.LoadImaged(keys=["image"]),  # Load the image
                mt.EnsureChannelFirstd(
                    keys=["image"]
                ),  # Make sure that the data is channels first format
                # Threshold the hounsfield units to the training-specific range and normalize
                mt.ThresholdIntensityd(
                    keys=["image"],
                    threshold=task_config["percentile_995"],
                    above=False,
                    cval=task_config["percentile_995"],
                ),
                mt.ThresholdIntensityd(
                    keys=["image"],
                    threshold=task_config["percentile_005"],
                    above=True,
                    cval=task_config["percentile_005"],
                ),
                mt.NormalizeIntensityd(
                    keys=["image"],
                    subtrahend=task_config["mean"],
                    divisor=task_config["std"],
                ),
                # Crop the foreground to reduce size for inference
                mt.CropForegroundd(
                    keys=["image"],
                    source_key="image",
                    allow_smaller=True,
                    select_fn=lambda x: x > task_config["crop_threshold"],
                ),
                mt.Orientationd(
                    keys=["image"], axcodes="RAS"
                ),  # Ensure the orientation is RAS
                mt.Spacingd(
                    keys=["image"], pixdim=task_config["spacing"], mode="bilinear"
                ),
                mt.EnsureTyped(
                    keys=["image"], device=torch.device("cpu"), track_meta=True
                ),
            ]
        )
    else:
        # Define the preprocessing pipeline
        preprocessing = mt.Compose(
            [
                mt.LoadImaged(keys=["image"]),  # Load the image
                mt.EnsureChannelFirstd(
                    keys=["image"]
                ),  # Make sure that the data is channels first format
                # Threshold the hounsfield units to the training-specific range and normalize
                mt.ThresholdIntensityd(
                    keys=["image"],
                    threshold=task_config["percentile_995"],
                    above=False,
                    cval=task_config["percentile_995"],
                ),
                mt.ThresholdIntensityd(
                    keys=["image"],
                    threshold=task_config["percentile_005"],
                    above=True,
                    cval=task_config["percentile_005"],
                ),
                mt.NormalizeIntensityd(
                    keys=["image"],
                    subtrahend=task_config["mean"],
                    divisor=task_config["std"],
                ),
                # Crop the foreground to reduce size for inference
                mt.CropForegroundd(
                    keys=["image"],
                    source_key="image",
                    allow_smaller=True,
                    select_fn=lambda x: x > task_config["crop_threshold"],
                ),
                mt.Orientationd(
                    keys=["image"], axcodes="RAS"
                ),  # Ensure the orientation is RAS
                mt.Spacingd(
                    keys=["image"], pixdim=task_config["spacing"], mode="bilinear"
                ),
                mt.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
                mt.EnsureTyped(
                    keys=["image"], device=torch.device("cpu"), track_meta=True
                ),
            ]
        )
    # This is the order of ops for either cuda or cpu. Will set device later
    postprocessing = mt.Compose(
        [
            mt.Activationsd(keys=["pred"], softmax=True),
            mt.AsDiscreted(keys=["pred"], argmax=True),
            mt.Invertd(
                keys=["pred"],
                transform=preprocessing,
                orig_keys=["image"],
                meta_keys="image_meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
            mt.SqueezeDimd(keys=["pred"], dim=0),
            mt.ToNumpyd(keys=["pred"], dtype=np.uint8),
        ]
    )
    # Define CPU postprocessing
    cpu_postprocessing = mt.Compose(
        [
            mt.EnsureTyped(keys=["pred"], device=torch.device("cpu"), track_meta=True),
            postprocessing,
        ]
    )
    # Define default postprocessing
    if device.type == "cpu":
        # In the case where the entire pipeline is being run on cpu, default_postprocessing = cpu_postprocessing
        default_postprocessing = cpu_postprocessing
    else:
        # GPU postprocessing
        default_postprocessing = mt.Compose(
            [
                mt.EnsureTyped(keys=["pred"], device=device, track_meta=True),
                postprocessing,
            ]
        )
    return preprocessing, cpu_postprocessing, default_postprocessing


def _create_dataloader(
    nifti_data: list[dict],
    preprocessing_transform: mt.Compose,
    num_dataloader_workers: int,
) -> ThreadDataLoader:
    """Creates a ThreadDataLoader for running inference
    :param nifti_data: a list of dictionaries in monai format containing nifti paths
    :param preprocessing_transform: the preprocessing function generated by _create_transforms
    :param num_dataloader_workers: the number of workers to load the data
    :return: a ThreadDataLoader object to use for inference
    """
    # We use CacheDataset for faster inference time - holds all passed images in RAM
    # Also helps with I/O if using multiple GPUS
    dataset = ErrorHandlingCacheDataset(
        data=nifti_data,
        transform=preprocessing_transform,
        cache_rate=1.0,
        num_workers=num_dataloader_workers,
        copy_cache=False,
    )
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=_remove_bad_cache_loads,
    )
    return dataloader


def _load_model(task: str, device: torch.device) -> torch.nn.Module:
    """Load model from disk to be used for inference
    :param task: Task name ['total', 'tissues', 'body']
    :param device: Device type ['cpu', 'cuda']
    :return: Loaded model
    """
    if task == "total":
        # Total model architecture needs to be defined
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=67,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(1, 2, 2, 2, 2),
            act=("LeakyReLU", {"inplace": True, "negative_slope": 0.01}),
            norm=("instance", {"affine": True}),
            num_res_units=2,
            adn_ordering="NAD",
        ).to(device)
        model.load_state_dict(
            torch.load(f"{MODELS_DIR}/torch_{task}.pth", map_location=device)
        )
    else:
        # The other two weights include the full model architecture as well
        model = torch.load(f"{MODELS_DIR}/torch_{task}.pt", map_location=device).to(
            device
        )
    # Set the model to eval
    model.eval()
    return model


def _reset_preprocessing(preprocessing: mt.Compose) -> mt.Compose:
    """A utility function to reset the preprocessing function due to monai's dumb behavior
    :param preprocessing: the original preprocessing function
    :return: the reset preprocessing function
    """
    preprocessing.tracing = True
    for transform in preprocessing.transforms:
        transform.tracing = True
    if hasattr(preprocessing.transforms[-2], "spacing_transform"):
        preprocessing.transforms[-2].spacing_transform.sp_resample.tracing = True
    if hasattr(preprocessing.transforms[-3], "spacing_transform"):
        preprocessing.transforms[-3].spacing_transform.sp_resample.tracing = True
    return preprocessing


def _write_segmentation_to_file(
    data: dict, image_meta_dict: dict, task: str
) -> str | None:
    """Write the predicted segmentation to a folder next to the original image
    :param data: the monai data dict
    :param image_meta_dict: the monai generated image meta information
    :param task: the task name ['total', 'tissues', 'body']
    :return: a string containing the output file name
    """
    # Define the nifti format image
    out_img = nib.Nifti1Image(
        data["pred"],  # This is a numpy array due to our postprocessing transforms
        affine=image_meta_dict["original_affine"].squeeze().numpy(),
        dtype=np.uint8,  # use the smallest integer type possible to save storage space
    )
    # Create the output directory if it does not exist
    original_name = image_meta_dict["filename_or_obj"][0]
    output_dir = original_name.replace(".nii.gz", "_segs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Make the output name for the file
    nii_file_name = os.path.basename(original_name).replace(".nii.gz", "")
    output_file = f"{output_dir}/{nii_file_name}_{task}.nii.gz"
    try:
        nib.save(out_img, output_file)
        return output_file
    except Exception as e:
        print(format_exc())
        logger.error(
            f"Writing segmentation for {original_name} failed due to {type(e).__name__}: {str(e)}",
            extra={
                "key": "segmentation",
                "input_nifti": original_name,
                "task": task,
                "completed": False,
                "failed_reason": type(e).__name__,
                "output_nifti": None,
                "inference_time": None,
                "processing_time": None,
            },
        )
        return


class ErrorHandlingCacheDataset(CacheDataset):
    # Custom load cache item to handle errors when loading data
    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        try:
            item = self.data[idx]
            first_random = self.transform.get_index_of_first(
                lambda t: isinstance(t, mt.RandomizableTrait)
                or not isinstance(t, mt.Transform)
            )
            item = self.transform(item, end=first_random, threading=True)

            if self.as_contiguous:
                item = mt.convert_to_contiguous(
                    item, memory_format=torch.contiguous_format
                )
            if item["image"].shape[0] != 1:
                nifti_name = self.data[idx]["image"]
                logger.error(
                    f"{nifti_name} is not a single channel image.",
                    extra={
                        "key": "load_seg",
                        "input_nifti": nifti_name,
                        "completed": False,
                        "failed_reason": "not single channel",
                        "error_message": None,
                    },
                )
                return {
                    "image": MetaTensor(
                        torch.zeros([1, 10, 10, 10]), meta={"skip": True}
                    )
                }
            return item
        except Exception as e:
            nifti_name = self.data[idx]["image"]
            logger.error(
                f"Loading {nifti_name} failed due to {type(e).__name__}: {e}",
                extra={
                    "key": "load_seg",
                    "input_nifti": nifti_name,
                    "completed": False,
                    "failed_reason": type(e).__name__,
                    "error_message": str(e),
                },
            )
            return {
                "image": MetaTensor(torch.zeros([1, 10, 10, 10]), meta={"skip": True})
            }


def _remove_bad_cache_loads(batch):
    """Remove any loads from the dataloader that are None
    :param batch: the batch from ErrorHandlingCacheDataset
    :return: the batch for inference
    """
    batch = [b for b in batch if b is not None]
    return list_data_collate(batch)
