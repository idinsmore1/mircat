# Module for curved planar reformation from centerline. Only good for straightened CPR currently
import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline, interpn
from skimage import measure
from skimage.measure._regionprops import RegionProperties
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes


###### CPR Generation Functions ###########
def create_straightened_cpr(
    vessel: np.ndarray,
    centerline: np.ndarray,
    cross_section_xy: tuple,
    resolution: int = 1,
    sigma: int = 2,
    is_binary: bool = True,
) -> np.ndarray:
    """Create a straightened Curved Planar Reformation of a vessel using a centerline
    :param vessel: the vessel as a numpy array
    :param centerline: the centerline as a numpy array
    :param cross_section_xy: the number of points in (x, y) for the cross-section;
        i.e. (100, 100) will result in an extracted cross-section of shape (100, 100)
    :param resolution: the resolution to sample the cpr. Default = 1
    :param sigma: the sigma value for the gaussian filter. Default = 2
    :return: a numpy array containing the straightened curved planar reformation
    """
    tangents = _compute_tangent_vectors(centerline)
    cpr = []
    num_touching = 0
    for center_point, normal_vector in zip(centerline, tangents):
        cross_section = _extract_cross_sectional_slice(
            vessel, center_point, normal_vector, cross_section_xy, resolution, is_binary
        )
        if is_binary:
            cross_section = _postprocess_cross_section(cross_section, sigma)
        cpr.append(cross_section)
    if num_touching > 0:
        logger.debug(f"Number of border touching cross-sections dropped: {num_touching}")
    cpr = np.stack(cpr, axis=0)
    return cpr


def _compute_tangent_vectors(centerline: np.ndarray) -> np.ndarray:
    """Compute the tangent vectors for a given centerline using a CubicSpline. Assumes that vertices are in natural order
    :param centerline: the centerline as an (N, 3) array
    :return: the tangent vectors for each vertex in the centerline
    """
    # Find the spline points
    s = np.zeros(len(centerline))
    s[1:] = np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0) ** 2, axis=1)))
    # Generate the spline
    cs = CubicSpline(s, centerline, bc_type="natural")
    # Compute and normalize the tangents
    tangents = cs(s, 1)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return tangents


def _compute_orthogonal_vectors(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find two orthogonal vectors to the original vector.
    :param vector: the vector to find the orthogonal vectors
    :return: the orthogonal vectors
    """
    if vector[0] == 0 and vector[1] == 0:
        if vector[2] == 0:
            # vec is a zero vector
            return None, None
        # vec is along the z-axis
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    else:
        v1 = np.array([-vector[1], vector[0], 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(vector, v1)
        return v1, v2


def _extract_cross_sectional_slice(arr, point, tangent_vector, slice_size, resolution, is_binary):
    """
    Extract a 2D cross-sectional slice from a 3D array given a point and its tangent vector.

    :param arr: 3D binary NumPy array
    :param point: The point in the array (x, y, z)
    :param tangent_vector: The tangent vector at the point
    :param slice_size: The size of the slice to extract (width, height)
    :param resolution: The resolution of the slice (spacing between points in the grid)
    :param is_binary: Is the array binary or not
    """
    # Find two orthogonal vectors
    v1, v2 = _compute_orthogonal_vectors(tangent_vector)
    if v1 is None or v2 is None:
        raise ValueError("Invalid tangent vector provided.")
    # Generate a meshgrid for the slice
    width, height = slice_size
    x_lin = np.linspace(-width / 2, width / 2, int(width / resolution))
    y_lin = np.linspace(-height / 2, height / 2, int(height / resolution))
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    # Map the grid points back to the 3D array indices
    slice_points = point + x_grid[..., np.newaxis] * v1 + y_grid[..., np.newaxis] * v2
    x = np.arange(arr.shape[0])
    y = np.arange(arr.shape[1])
    z = np.arange(arr.shape[2])
    if is_binary:
        # Perform nearest neighbor interpolation in each axis and fill with 0
        method = 'nearest'
        fill_value = 0
    else:
        # Perform linear interpolation in each axis and fill with min value
        method = "linear"
        fill_value = arr.min()
    # Perform the interpolation
    slice_2d = interpn(
        (x, y, z),
        arr,
        slice_points,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )
    return slice_2d


def _postprocess_cross_section(cross_section: np.ndarray, sigma: int) -> tuple[np.ndarray, int]:
    """Fill small gaps within the cross-section and apply a gaussian filter to smooth the edges
    :param cross_section: the extracted cross-section from _extract_cross_sectional_slice
    :param sigma: the gaussian kernel sigma
    :return: the smoothed cross-section
    """
    cross_section_labels = [label for label in np.unique(cross_section) if label > 0]
    if len(cross_section_labels) == 0:
        return cross_section
    output_cross_section = np.zeros_like(cross_section, dtype=np.uint8)
    for label in cross_section_labels:
        center_label = _filter_label_regions(cross_section, label)
        center_label = remove_small_holes(center_label)
        center_label = gaussian(center_label, sigma=sigma).round(0)
        # Assign the output of the binary to the appropriate label
        output_cross_section[center_label == 1] = label  
    # # Check if the center label touches the image border
    # height, width = output_cross_section.shape
    # main_label = (output_cross_section == 1).astype(int)
    # if np.any(main_label):
    #     borders = {"top": 0, "bottom": height - 1, "left": 0, "right": width - 1}
    #     touches_border = {
    #         "top": np.any(main_label[borders["top"], :]),
    #         "bottom": np.any(main_label[borders["bottom"], :]),
    #         "left": np.any(main_label[:, borders["left"]]),
    #         "right": np.any(main_label[:, borders["right"]]),
    #     }
    #     touching_any = any(touches_border.values())
    #     if touching_any:
    #         return np.zeros_like(cross_section), 1
    #     main_region = _get_regions(main_label)
    #     solidity = main_region[0].solidity
    #     print(solidity)
    return output_cross_section


def _filter_label_regions(cross_section: np.ndarray, label: int) -> np.ndarray:
    """Filter the labels within a CPR cross-section to only return the label region closest to the center of the image
    :param cross_section: the 2d numpy array containing the entire cross-section
    :param label: the label to filter
    :return: a numpy array containing the filtered cross-section
    """
    tmp_cross_section = cross_section.copy()  # Create a temporary array, so we don't change the original cross-section
    tmp_cross_section[tmp_cross_section != label] = 0  # set anything that is not the desired label to 0
    tmp_cross_section[tmp_cross_section == label] = 1  # set the label to 1 for easier replacement later
    regions = _get_regions(tmp_cross_section)
    if len(regions) == 0:
        centered_label = np.zeros_like(cross_section)
    elif len(regions) > 1:
        centered_label = _closest_to_centroid(tmp_cross_section, regions)
    else:
        centered_label = tmp_cross_section
    return centered_label > 0


def _closest_to_centroid(cross_section: np.ndarray, regions: list[RegionProperties]) -> np.ndarray:
    """Filter a cross-section label using skimage.measure.regionprops to the region closest to the center
    :param cross_section: the cross-section array
    :param regions: the list output from skimage.measure.regionprops
    :return: the filtered numpy array
    """
    center_of_plane = np.array(cross_section.shape) / 2.0
    centroids = [np.array(region.centroid) for region in regions]
    distance_per_region = np.asarray([np.linalg.norm(centroid - center_of_plane) for centroid in centroids])
    min_distance_region_idx = int(np.argmin(distance_per_region))
    center_region = regions[min_distance_region_idx]
    center_label = np.zeros_like(cross_section)
    center_label[center_region.coords[:, 0], center_region.coords[:, 1]] = 1
    return center_label


######### CPR Measurement Functions #############
def measure_largest_cpr_diameter(cpr: np.ndarray, pixel_spacing: tuple, diff_threshold: int = 5) -> dict:
    """Find the largest average cpr diameter
    :param cpr: a binary array containing the entire curved planar reformation to be measured
    :param pixel_spacing: the x, y pixel spacing for each cross-section
    :param diff_threshold: the maximum difference allowed between major and minor diameters for a cross-section.
        If |major - minor| > diff_threshold, set the average diameter = minor diameter to be conservative. Default = 5
    :return: a dictionary with {'avg_diam', 'major_diam', 'minor_diam'} as keys,
        holding the data from the largest diameter cross-section
    """
    avg_diams = []
    major_diams = []
    minor_diams = []
    solidity_threshold = 0.95
    circularity_threshold = 0.9
    eccentricity_threshold = 0.8
    for cross_section in cpr:
        data = measure_cross_sectional_diameter(
            cross_section, pixel_spacing, diff_threshold
        )
        is_convex = data['solidity'] >= solidity_threshold
        is_circular = data['circularity'] >= circularity_threshold and data['circularity'] <= 1.1
        is_not_elliptical = data['eccentricity'] <= eccentricity_threshold
        if is_convex and is_circular and is_not_elliptical:
            avg_diam = data['max_diam']
            major_diam = data['major_diam']
            minor_diam = data['minor_diam']
            avg_diams.append(avg_diam)
            major_diams.append(major_diam)
            minor_diams.append(minor_diam)
    if avg_diams:
        largest_idx = np.argmax(avg_diams)
        diam_dict = {
            "avg_diam": avg_diams[largest_idx],
            "major_diam": major_diams[largest_idx],
            "minor_diam": minor_diams[largest_idx],
        }
    else:
        diam_dict = {}
    return diam_dict


def measure_mid_cpr_diameter(cpr: np.ndarray, pixel_spacing: tuple) -> dict:
    """Measure the mid-region diameter of a curved planar reformation
    :param cpr: a binary array containing the entire curved planar reformation to be measured
    :param pixel_spacing: the x, y pixel spacing for each cross-section
    :return: a dictionary with the mid-region average diameter"""
    len_cpr = len(cpr)
    midslice = len_cpr // 2
    if len_cpr > 0:
        data = measure_cross_sectional_diameter(cpr[midslice], pixel_spacing, diff_threshold=5)
        avg_diam = data["max_diam"]
    else:
        avg_diam = None
    return {"mid_diam": avg_diam}


def measure_cross_sectional_diameter(
    cross_section: np.ndarray, pixel_spacing: tuple, diff_threshold: int
) -> dict[str, float]:
    """Measure the cross-sectional diameter from a straightened cpr slice
    :param cross_section: the binary straightened cpr slice as a numpy array
    :param pixel_spacing: the pixel spacing of the cpr slice
    :param diff_threshold: the maximum difference allowed between major and minor diameters.
        If |major - minor| > diff_threshold, set the average diameter = minor diameter to be conservative
    :return: a dictionary containing the max, major, minor diameters, as well as the solidarity, cicularity, and eccentricity
    """
    regions = _get_regions(cross_section)
    data = {
        'max_diam': 0,
        'major_diam': 0,
        'minor_diam': 0,
        'solidity': 0,
        'circularity': 0,
        'eccentricity': 1.0
    }
    if len(regions) == 0:
        return data
    region = regions[0]
    major_endpoints, minor_endpoints = _get_cross_section_endpoints(region)
    major_units = list(np.multiply(major_endpoints, pixel_spacing))
    minor_units = list(np.multiply(minor_endpoints, pixel_spacing))
    major_diam = _endpoint_euclidean_distance(major_units)
    minor_diam = _endpoint_euclidean_distance(minor_units)
    if abs(major_diam - minor_diam) < diff_threshold:
        max_diam = round((major_diam + minor_diam) / 2, 1)
    else:
        max_diam = min(major_diam, minor_diam)
    solidity = region.solidity 
    circularity = (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0
    eccentricity = region.eccentricity
    data.update({
        'max_diam': max_diam,
        'major_diam': major_diam,
        'minor_diam': minor_diam,
        'solidity': round(solidity, 2),
        'circularity': round(circularity, 2),
        'eccentricity': round(eccentricity, 2)
    })
    return data


def _get_regions(image: np.ndarray) -> list[RegionProperties]:
    """Get the regions of an image using skimage.measure.regionprops
    :param image: the input image
    :return: the regions of an image
    """
    labels = measure.label(image)
    regions = measure.regionprops(labels)
    return regions


def _get_cross_section_endpoints(
    region: RegionProperties,
) -> tuple[tuple[tuple, tuple], tuple[tuple, tuple]]:
    centroid = region.centroid
    orientation = region.orientation
    major_endpoints = _get_axis_endpoints(centroid, orientation, region.axis_major_length)
    minor_endpoints = _get_axis_endpoints(centroid, orientation, region.axis_minor_length)
    return major_endpoints, minor_endpoints


def _get_axis_endpoints(centroid: np.array, orientation: float, axis_length: float) -> tuple[tuple, tuple]:
    """Calculate the endpoints of the axis of a cross-section region
    :param centroid: the region.centroid
    :param orientation: the region.orientation
    :param axis_length: region.axis_major_length or region.axis_minor_length
    :return: a tuple containing the endpoints
    """
    y0, x0 = centroid
    # calculate the endpoints of the major axis using the centroid
    x1 = x0 - np.sin(orientation) * 0.5 * axis_length
    x2 = x0 + np.sin(orientation) * 0.5 * axis_length
    y1 = y0 - np.cos(orientation) * 0.5 * axis_length
    y2 = y0 + np.cos(orientation) * 0.5 * axis_length
    return (y1, x1), (y2, x2)


def _endpoint_euclidean_distance(endpoints: list) -> float:
    """Calculate the Euclidean distance between endpoints
    :param endpoints: the endpoints (must be two of them)
    :return: the Euclidean distance
    """
    p0, p1 = [*endpoints]
    return round(np.sqrt(((p0 - p1) ** 2).sum()), 1)
