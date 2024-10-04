# Module for Centerline creation for vessels

import numpy as np

from loguru import logger
from kimimaro import skeletonize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from mircat_stats.statistics.cpr import _compute_tangent_vectors

# These are the allowed arguments for kimimaro.skeletonize()
# These are specifically for the teasar_params dictionary argument
TEASAR_KWARGS = {
    "scale",
    "const",
    "pdrf_scale",
    "pdrf_exponent",
    "soma_acceptance_threshold",
    "soma_detection_threshold",
    "soma_invalidation_const",
    "soma_invalidation_scale",
    "max_paths",
}
# These are the rest of the arguments
NON_TEASAR_KWARGS = {
    "dust_threshold",
    "progress",
    "fix_branching",
    "in_place",
    "fix_borders",
    "parallel",
    "parallel_chunk_size",
    "extra_targets_before" "extra_targets_after",
    "fill_holes",
    "fix_avocados",
    "voxel_graph",
}
SKELETONIZE_KWARGS = TEASAR_KWARGS.union(NON_TEASAR_KWARGS)
SAMPLING_KWARGS = {"is_thoracic", "number_of_points", "window_length"}


def create_centerline(
    vessel: np.ndarray, voxel_spacing: tuple, vessel_label: int = 1, **kwargs
) -> np.ndarray | None:
    """Create a centerline for a given vessel using skeletonize_vessel and postprocess_skeleton
    :param vessel: vessel array
    :param voxel_spacing: voxel spacing
    :param vessel_label: vessel label
    :param kwargs: kwargs for skeletonize_vessel and postprocess_skeleton
    """
    skeleton_kwargs, sampling_kwargs = _validate_centerline_kwargs(kwargs)
    vertices, edges = skeletonize_vessel(
        vessel, voxel_spacing, vessel_label, **skeleton_kwargs
    )
    if vertices is None:
        return None
    centerline = postprocess_skeleton(vertices, edges, **sampling_kwargs)
    return centerline


def skeletonize_vessel(
    vessel: np.ndarray, voxel_spacing: tuple, vessel_label: int = 1, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the raw skeleton of the vessel. Returns skeleton vertices and edges
    :param vessel: the vessel numpy array
    :param voxel_spacing: the spacing of voxels in the array
    :param vessel_label: the label of the vessel in the array, default=1 (binary image)
    :param kwargs: additional keyword arguments for kimimaro.skeletonize() function
    :return: a tuple containing the centerline vertices and edges
    """
    # These are the 2 necessary parameters for teasar algorithm
    teasar_params = {"const": 40, "scale": 1.5}
    if kwargs:
        _validate_skeletonize_kwargs(kwargs)
        teasar_kwargs = _extract_teasar_kwargs(kwargs)
        teasar_params.update(teasar_kwargs)  # Update or add more teasar params
    # for each parameter, get the kwarg value, if it doesn't exist, use the default from the function
    skeleton = skeletonize(
        all_labels=vessel,
        teasar_params=teasar_params,
        anisotropy=voxel_spacing,
        object_ids=[vessel_label],
        dust_threshold=kwargs.get("dust_threshold", 1000),
        progress=kwargs.get("progress", False),
        fix_branching=kwargs.get("fix_branching", True),
        in_place=kwargs.get("in_place", False),
        fix_borders=kwargs.get("fix_borders", True),
        parallel=kwargs.get("parallel", 1),
        parallel_chunk_size=kwargs.get("parallel_chunk_size", 100),
        extra_targets_before=kwargs.get("extra_targets_before", []),
        extra_targets_after=kwargs.get("extra_targets_after", []),
        fill_holes=kwargs.get("fill_holes", False),
        fix_avocados=kwargs.get("fix_avocados", False),
        voxel_graph=kwargs.get("voxel_graph", None),
    )
    try:
        skel = skeleton[vessel_label]
        vertices = (
            skel.vertices / voxel_spacing
        )  # Divide by voxel spacing to have the units match
        edges = skel.edges
        return vertices, edges
    except KeyError:
        return None, None


def postprocess_skeleton(
    vertices: np.ndarray, edges: np.ndarray, **kwargs
) -> np.ndarray:
    """Postprocess the kimimaro skeleton into a final centerline
    :param vertices: the vertices of the centerline
    :param edges: the edges listing the connections between vertices
    :param is_thoracic: is the skeleton of the thoracic aorta
    :param kwargs: Additional optional arguments. Must be in [is_thoracic, number_of_points, window_length]
    :return: the final centerline as an (N, 3) numpy array
    """
    skeleton_length = vertices.shape[0]
    _validate_sampling_kwargs(kwargs)
    num_points, window_length = _get_sampling_values(skeleton_length, kwargs)
    ordered_centerline = _order_vertices(vertices, edges)
    spaced_centerline = _arclength_space_centerline(ordered_centerline)
    even_sampled_centerline = _evenly_sample_centerline(spaced_centerline, num_points)
    smoothed_centerline = _smooth_centerline(even_sampled_centerline, window_length)
    try:
        _compute_tangent_vectors(smoothed_centerline)
    except ValueError:
        # If computing tangent vectors fails, recursively reduce the number of centerline points until it works
        new_num_points = int(np.floor(num_points * 0.9))
        new_window_length = window_length - 1
        if new_window_length > new_num_points:
            new_window_length = new_num_points - 1
        return postprocess_skeleton(
            vertices,
            edges,
            number_of_points=new_num_points,
            window_length=new_window_length,
        )
    return smoothed_centerline


def _order_vertices(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Order the vertices of the skeleton from end to end by using the edges
    :param vertices: the vertices of the centerline
    :param edges: the edges listing the connections between vertices
    :return: the ordered centerline in the same shape as the skeleton
    """
    # Step 1: Build the adjacency list
    adjacency_list = {}
    for edge in edges:
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        if edge[1] not in adjacency_list:
            adjacency_list[edge[1]] = []
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])
    # Step 2: Identify the start (and end) vertex as it will only have 1 edge
    start_vertex = None
    for vertex, connected in adjacency_list.items():
        if len(connected) == 1:
            start_vertex = vertex
            break
    # Sanity check
    if start_vertex is None:
        raise ValueError("A start vertex could not be found.")
    # Step 3: Traverse the graph from the start vertex
    ordered_vertices_indices = [start_vertex]
    current_vertex = start_vertex
    # Since we know the length of the path, we can loop N times
    for _ in range(len(vertices) - 1):
        # The next vertex will be the one that is not the previous
        for vertex in adjacency_list[current_vertex]:
            if vertex not in ordered_vertices_indices:
                ordered_vertices_indices.append(vertex)
                break
        current_vertex = ordered_vertices_indices[-1]
    # Step 4: Map the ordered indices to the original vertices
    ordered_vertices = [vertices[idx] for idx in ordered_vertices_indices]
    return np.asarray(ordered_vertices)


def _arclength_space_centerline(ordered_centerline: np.ndarray) -> np.ndarray:
    """evenly space the centerline in array space
    :param ordered_centerline: the output of _order_vertices
    :return: the evenly spaced centerline
    """
    differences = np.diff(ordered_centerline, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    arc_lengths = np.concatenate(([0], np.cumsum(distances)))
    total_length = arc_lengths[-1]
    normalized_arc_lengths = arc_lengths / total_length
    # Create interpolation functions for each coordinate
    interp_funcs = [
        interp1d(normalized_arc_lengths, ordered_centerline[:, i]) for i in range(3)
    ]
    # Create a new uniform parameterization
    uniform_t = np.linspace(0, 1, len(ordered_centerline))
    # Interpolate the centerline points
    uniform_centerline = np.column_stack([f(uniform_t) for f in interp_funcs])
    return uniform_centerline


def _evenly_sample_centerline(
    spaced_centerline: np.ndarray, number_of_points: int
) -> np.ndarray:
    """Evenly sample a specified number of points along the centerline
    :param spaced_centerline: the output of _arclength_space_centerline
    :param number_of_points: the number of points to sample from the centerline
    :return: the evenly sampled centerline of shape (number_of_points, 3)
    """
    # Calculate the cumulative distance along the centerline
    distances = np.cumsum(
        np.r_[0, np.linalg.norm(np.diff(spaced_centerline, axis=0), axis=1)]
    )
    total_length = distances[-1]

    # Create an interpolation function for each dimension
    f_z = interp1d(distances, spaced_centerline[:, 0])
    f_y = interp1d(distances, spaced_centerline[:, 1])
    f_x = interp1d(distances, spaced_centerline[:, 2])

    # Evenly sample distances along the centerline
    sample_points = np.linspace(0, total_length, number_of_points)
    # Sample the centerline
    sampled_centerline = np.column_stack(
        (f_z(sample_points), f_y(sample_points), f_x(sample_points))
    )
    return sampled_centerline


def _smooth_centerline(
    sampled_centerline: np.ndarray, window_length: int
) -> np.ndarray:
    """Use a window length to smooth the centerline using a savitzky-golay filter
    :param sampled_centerline: the sampled point centerline from _evenly_sample_centerline
    :param window_length: the length of the filter window
    :return: the smoothed centerline
    """
    n_dimensions = sampled_centerline.shape[1]
    polyorder = 2

    def _change_if_odd(window):
        if not window % 2:
            window += 1
        return window

    window_length = _change_if_odd(window_length)
    if window_length > len(sampled_centerline):
        window_length = _change_if_odd(int(len(sampled_centerline) // 2))
    if window_length < polyorder:
        logger.debug(
            f"Window length {window_length} is smaller than polyorder {polyorder}"
        )
        window_length = polyorder + 1
        # raise ValueError(f'Window length {window_length} is smaller than polyorder {polyorder}')
    smoothed_centerline = np.zeros_like(sampled_centerline)
    for dim in range(n_dimensions):
        smoothed_centerline[:, dim] = savgol_filter(
            sampled_centerline[:, dim], window_length, polyorder
        )
    smoothed_centerline = smoothed_centerline.round().astype(
        np.uint16
    )  # Make sure the centerline is integers
    return smoothed_centerline


def _get_sampling_values(skeleton_length: int, kwargs: dict) -> tuple[int, int]:
    """Get the number of points and window length for the sampling function using the original shape and kwargs
    :param skeleton_length: the original number of points in the skeleton, usually given as skeleton.shape[0]
    :param kwargs: a dictionary of keyword arguments with parameters necessary
    :return: a tuple containing the (number_of_points, window_length)
    """
    is_thoracic = kwargs.get("is_thoracic", False)
    has_specified_values = (
        kwargs.get("number_of_points") is not None
        and kwargs.get("window_length") is not None
    )
    if has_specified_values:
        number_of_points = kwargs.get("number_of_points")
        window_length = kwargs.get("window_length")
    elif is_thoracic:
        # For the thoracic aorta, we want fewer number of points and slightly wider window range
        number_of_points = int(skeleton_length // 2)
        window_length = int(number_of_points // 8)
    else:
        number_of_points = int(skeleton_length // 1.5)
        window_length = int(number_of_points // 10)
    return number_of_points, window_length


def _validate_centerline_kwargs(kwargs) -> tuple[dict, dict]:
    """Validate create_centerline keyword arguments and return the function specific arguments"""
    all_kwargs = SKELETONIZE_KWARGS.union(SAMPLING_KWARGS)
    assert kwargs == {} or all(k in all_kwargs for k in kwargs), ValueError(
        f"Invalid kwargs given to create_centerline: {kwargs}. Must be in {sorted(all_kwargs)}."
    )
    skeleton_kwargs = {k: v for k, v in kwargs.items() if k in SKELETONIZE_KWARGS}
    sampling_kwargs = {k: v for k, v in kwargs.items() if k in SAMPLING_KWARGS}
    return skeleton_kwargs, sampling_kwargs


def _validate_sampling_kwargs(kwargs):
    """Validate sampling keyword arguments"""
    assert kwargs == {} or all([k in SAMPLING_KWARGS for k in kwargs]), ValueError(
        f"Sampling kwargs must be in {SAMPLING_KWARGS}"
    )
    is_thoracic = kwargs.get("is_thoracic")
    number_of_points = kwargs.get("number_of_points")
    window_length = kwargs.get("window_length")
    # Check thoracic first
    if is_thoracic is not None:
        assert is_thoracic in [True, False], ValueError(
            f'is_thoracic must be a boolean: {kwargs.get("is_thoracic")=}'
        )
    if is_thoracic is None:
        assert number_of_points is not None and window_length is not None, ValueError(
            "number_of_points and window_length must be given if is_thoracic is None"
        )
    elif number_of_points is not None:
        assert isinstance(number_of_points, int), ValueError(
            "number_of_points must be an integer"
        )
        assert isinstance(window_length, int), ValueError(
            "window_length must be an integer provided with number_of_points"
        )
    elif window_length is not None:
        assert isinstance(window_length, int), ValueError(
            "window_length must be an integer"
        )
        assert isinstance(number_of_points, int), ValueError(
            "number_of_points must be an integer provided with window_length"
        )


def _validate_skeletonize_kwargs(kwargs: dict) -> None:
    """Validate skeletonize keyword arguments"""
    assert all([k in SKELETONIZE_KWARGS for k in kwargs]), ValueError(
        f"All kwargs for create_centerline must be in {SKELETONIZE_KWARGS}"
    )


def _extract_teasar_kwargs(kwargs: dict) -> dict:
    """Extract the teasar only kwargs"""
    return {key: value for key, value in kwargs.items() if key in TEASAR_KWARGS}
