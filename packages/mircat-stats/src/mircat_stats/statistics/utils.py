import SimpleITK as sitk
import numpy as np


def _calc_shape_stats(seg: sitk.Image) -> sitk.LabelShapeStatisticsImageFilter:
    """Calculate the shape stats for a segmentation
    Parameters
    ----------
    seg : sitk.Image
        The segmentation to calculate shape stats for
    Returns
    -------
    sitk.LabelShapeStatisticsImageFilter
        The executed shape stats
    """
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.SetGlobalDefaultCoordinateTolerance(1e-5)
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.ComputePerimeterOn()
    shape_stats.Execute(seg)
    return shape_stats


def _calc_intensity_stats(
    image: sitk.Image, seg: sitk.Image
) -> sitk.LabelIntensityStatisticsImageFilter:
    """Calculate intensity stats for a segmentation using the reference image
    Parameters
    ----------
    image : sitk.Image
        The reference image
    seg : sitk.Image
        The segmentation of the reference image
    Returns
    -------
    sitk.LabelIntensityStatisticsImageFilter
        The executed intensity stats
    """
    intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
    intensity_stats.SetGlobalDefaultCoordinateTolerance(1e-5)
    intensity_stats.Execute(seg, image)
    return intensity_stats


def _calculate_3d_volumes_and_intensities(
    image: sitk.Image,
    seg: sitk.Image,
    output_map: dict,
    prefix: str = "",
    endpoints: tuple = (0, None),
) -> dict:
    """Calculate the total volume and average intensity for structures in a region of 3d image
    :param image: the original image
    :param seg: the segmentation of the image
    :param output_map: the dictionary containing the output segmentation label map
    :param prefix: prefix to the statistics name, default is empty ''. Useful if measuring a specific region
    :param endpoints: Specific z-axis endpoints for a region of interest in the shape (start, end). Default is (0, None),
        which will measure the entire image. If end = None, will measure from start_idx -> rest of the scan
    :return: a dictionary containing the volume in cm3 and average intensity for each segmentation in the output map
        found in the image. If the segmentation does not exist in the image, the return value will be None
    """
    volumes = _calculate_3d_volumes(seg, output_map, prefix, endpoints)
    intensities = _calculate_3d_intensities(image, seg, output_map, prefix, endpoints)
    if len(volumes) == len(intensities):
        stats = {}
        for (vol_key, vol_val), (inten_key, inten_val) in zip(
            volumes.items(), intensities.items()
        ):
            ord_stats = {vol_key: vol_val, inten_key: inten_val}
            stats.update(ord_stats)
    else:
        stats = {**volumes, **intensities}
    return stats


def _calculate_3d_volumes(
    seg: sitk.Image, output_map: dict, prefix: str = "", endpoints: tuple = (0, None)
) -> dict:
    """Calculate the total volume for each segmentation in a sitk image
    :param seg: the sitk image containing the segmentation
    :param output_map: the dictionary containing the output segmentation label map
    :param prefix: prefix to the statistics name, default is empty ''. Useful if measuring a specific region
    :param endpoints: Specific z-axis endpoints for a region of interest in the shape (start, end). Default is (0, None),
        which will measure the entire image. If end = None, will measure from start_idx -> rest of the scan
    :return: a dictionary containing the volumes for each segmentation
    """
    seg = _slice_images(endpoints, seg)  # type: ignore
    shape_stats = _calc_shape_stats(seg)
    seg_labels = (
        shape_stats.GetLabels()
    )  # This is all labels that were found in the image
    volumes = {}
    for name, label in output_map.items():
        volume = None
        if label in seg_labels:
            volume = round(
                shape_stats.GetPhysicalSize(label) / 1000, 1
            )  # This gives the volume in cm3 for CT
        volumes[f"{prefix}{name}_volume_cm3"] = volume
    return volumes


def _calculate_3d_intensities(
    image: sitk.Image,
    seg: sitk.Image,
    output_map: dict,
    prefix: str = "",
    endpoints: tuple = (0, None),
) -> dict:
    """Calculate the average intensity for each segmentation in the reference image
    :param image: the reference image
    :param seg: the segmentation image
    :param output_map: the dictionary containing the output segmentation label map
    :param prefix: prefix to the statistics name, default is empty ''. Useful if measuring a specific region
    :param endpoints: Specific z-axis endpoints for a region of interest in the shape (start, end). Default is (0, None),
        which will measure the entire image. If end = None, will measure from start_idx -> rest of the scan
    :return: a dictionary containing the average intensities for each segmentation
    """
    image, seg = _slice_images(endpoints, [image, seg])
    intensity_stats = _calc_intensity_stats(image, seg)
    seg_labels = intensity_stats.GetLabels()
    intensities = {}
    for name, label in output_map.items():
        intensity = None
        if label in seg_labels:
            intensity = round(intensity_stats.GetMean(label), 2)
        intensities[f"{prefix}{name}_average_intensity"] = intensity
    return intensities


def _calculate_2d_areas(
    seg: sitk.Image,
    output_map: dict,
    slice_idx: int,
    prefix: str,
    get_perimeter: bool = False,
) -> dict:
    """Calculate the area of each label in the segmentation within a 2d slice
    :param seg: the segmentation
    :param output_map: the label map
    :param slice_idx: the slice of the segmentation to measure
    :param prefix: the label of the slice (L1, L3, etc)
    :param get_perimeter: Measure the perimeters from the shape stats
    :return: a dictionary containing the area of each label found within the segmentation
    """
    seg_slice = seg[:, :, slice_idx]
    shape_stats = _calc_shape_stats(seg_slice)
    seg_labels = shape_stats.GetLabels()
    areas = {}
    for name, label in output_map.items():
        area = None
        if label in seg_labels:
            area = round(
                shape_stats.GetPhysicalSize(label) / 100, 1
            )  # This will be area in cm2 for CT
        areas[f"{prefix}{name}_area_cm2"] = area
        if get_perimeter and (label in seg_labels):
            # Note that the shape touches the border if at least 5 percent of the perimeter is on the border
            border_ratio = shape_stats.GetPerimeterOnBorderRatio(label) * 100
            raw_perim = (
                shape_stats.GetPerimeter(label) / 10
            )  # This will be perimeter in cm
            ellipse_perim = _calc_ellipsoid_perimeter(
                shape_stats.GetEquivalentEllipsoidDiameter(label)
            )
            circ_perim = shape_stats.GetEquivalentSphericalPerimeter(label) / 10
            areas[f"{prefix}{name}_border_ratio"] = round(border_ratio, 1)
            areas[f"{prefix}{name}_total_perimeter_cm"] = round(raw_perim, 1)
            areas[f"{prefix}{name}_ellipse_perimeter_cm"] = round(ellipse_perim, 1)
            areas[f"{prefix}{name}_circle_perimeter_cm"] = round(circ_perim, 1)
    return areas


def _slice_images(endpoints: tuple, images: sitk.Image | list) -> sitk.Image | list:
    """Slice any number of matching images to the same region using endpoints
    :param endpoints: Specific z-axis endpoints for a region of interest in the shape (start, end). Default is (0, None),
        which will measure the entire image. If end = None, will measure from start_idx -> rest of the scan
    :param images: a single sitk.Image or list of sitk.Images -> must be the same shape or error is raised
    """
    start, end = endpoints
    is_single_image = isinstance(images, sitk.Image)
    if end is None:
        if is_single_image:
            end = images.GetSize()[
                -1
            ]  # This is the full length of the image along the z-axis
        else:
            # Check that all images are the same size
            image_sizes = [img.GetSize() for img in images]
            assert image_sizes.count(image_sizes[0]) == len(image_sizes), ValueError(
                "All images in list must be the same shape"
            )
            end = images[0].GetSize()[-1]  # All images are assumed to be the same size
    if is_single_image:
        images = images[:, :, start:end]
        return images
    else:
        sliced_images = [image[:, :, start:end] for image in images]
        return sliced_images


def _calc_ellipsoid_perimeter(diameters: tuple) -> float:
    """Calculate the estimated ellipsoid diameters from a shape_stats.GetEquivalentEllipsoidDiameter result.
    Uses the Ramanujan estimation.
    :param diameters: the result from shape_stats
    :return the perimeter in cm
    """
    minor, major = diameters
    major /= 2
    minor /= 2
    perim = np.pi * (
        3 * (major + minor) - np.sqrt((3 * major + minor) * (major + 3 * minor))
    )
    return perim / 10  # This will be in cm


def _filter_largest_components(image, labels_of_interest):
    """Filter the largest connected component for each label in the image
    :param image: the input image
    :param labels_of_interest: the labels to filter
    :return: a new image with only the largest connected component for each label
    """
    # Initialize the output image
    output_image = sitk.Image(image.GetSize(), image.GetPixelID())
    output_image.CopyInformation(image)

    for label in labels_of_interest:
        # Create a binary mask for the current label
        binary_mask = sitk.BinaryThreshold(
            image, lowerThreshold=label, upperThreshold=label
        )

        # Find connected components
        connected_components = sitk.ConnectedComponent(binary_mask)

        # Get label statistics
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(connected_components)

        # Find the label of the largest component
        largest_label = max(
            label_stats.GetLabels(), key=lambda i: label_stats.GetPhysicalSize(i)
        )

        # Create a mask of the largest component
        largest_component = sitk.BinaryThreshold(
            connected_components,
            lowerThreshold=largest_label,
            upperThreshold=largest_label,
        )

        # Add this component to the output image
        output_image = sitk.Add(output_image, sitk.Multiply(largest_component, label))

    return output_image
