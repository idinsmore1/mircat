import numpy as np

from loguru import logger
from kimimaro import skeletonize
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d


class Centerline:
    # These are specifically for the teasar_params dictionary argument
    teasar_kwargs = {
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
    base_teasar_kwargs = {'scale': 1.0, 'const': 50}
    # These are the rest of the arguments
    non_teasar_kwargs = {
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
    base_non_teasar_kwargs = {
        "dust_threshold": 1000,
        "progress": False,
        "fix_branching": True,
        "in_place": True,
        "fix_borders": True,
        "parallel": 1,
        "parallel_chunk_size": 100,
        "extra_targets_before": [],
        "extra_targets_after": [],
        "fill_holes": False,
        "fix_avocados": False,
        "voxel_graph": None
    }
    skeleletonize_kwargs = teasar_kwargs.union(non_teasar_kwargs)

    @staticmethod
    def _validate_centerline_kwargs(kwargs) -> tuple[dict, dict]:
        """Validate the keyword arguments passed to the initialization of the Centerline object and
        split passed keywords into teasar and non-teasar arguments
        Parameters:
        -----------
        kwargs : dict
            The keyword arguments to validate
        Returns:
        --------
        tuple[dict, dict]
            The teasar and non-teasar arguments
        """
        assert all(k in Centerline.skeleletonize_kwargs for k in kwargs) or kwargs == {}, ValueError(
            f"Invalid kwargs given to create_centerline: {kwargs}. Must be in {sorted(Centerline.skeleletonize_kwargs)}."
        )
        teasar_kwargs = {k: v for k, v in kwargs.items() if k in Centerline.teasar_kwargs}
        non_teasar_kwargs = {k: v for k, v in kwargs.items() if k in Centerline.non_teasar_kwargs}
        return teasar_kwargs, non_teasar_kwargs
    
    def _set_centerline_kwargs(self, kwargs: dict) -> None:
        """Set the teasar and non-teasar keyword arguments for the skeletonize function
        Parameters:
        -----------
        kwargs : dict
            The passed keyword arguments to __init__ method
        non_teasar_kwargs : dict
            The non-teasar arguments
        Returns:
        --------
        None - sets the teasar_kwargs and non_teasar_kwargs attributes
        """
        teasar_kwargs, non_teasar_kwargs = self._validate_centerline_kwargs(kwargs)
        self.teasar_kwargs = Centerline.base_teasar_kwargs.copy()
        self.teasar_kwargs.update(teasar_kwargs)
        self.non_teasar_kwargs = Centerline.base_non_teasar_kwargs.copy()
        self.non_teasar_kwargs.update(non_teasar_kwargs)

    def __init__(self, spacing: tuple[float, float, float], label: int=1, **kwargs) -> np.ndarray | None:
        """Initialize a Centerline object
        Parameters:
        -----------
        spacing : tuple[float, float, float]
            The spacing of the image
        label : int
            The label to use for the centerline
        kwargs : dict
            The keyword arguments to pass to the skeletonize function
        """
        self.spacing = spacing
        self.label = label
        self.skeleton: tuple[np.ndarray, np.ndarray] | None = None
        self._set_centerline_kwargs(kwargs)


    def create_centerline(self, segmentation: np.ndarray, **kwargs) -> None:
        """Create a centerline on the segmentation and set it as self.centerline
        Parameters:
        -----------
        segmentation : np.ndarray
            The segmentation to create the centerline on
        **kwargs:
            min_points: int
                The minimum number of points to keep in the centerline. Default = 25
            max_points: int
                The maximum number of points to keep in the centerline. Default = 200
            smoothing_factor: float
                B-spline smoothing factor (0-1). Default = 0.5
            gaussian_sigma: float
                The sigma value for the gaussian smoothing. Default = 1.0
        """
        self._fit(segmentation)
        if self.skeleton is None:
            self.succeeded = False
            return
        try:
            self._postprocess_skeleton(**kwargs)
        except Exception as e:
            logger.error(f"Error postprocessing centerline: {e}")
            self.succeeded = False
        

    def _fit(self, segmentation: np.ndarray) -> None:
        """Fit a centerline on the segmentation
        Parameters:
        -----------
        segmentation : np.ndarray
            The segmentation to fit the centerline to
        """
        skeleton = skeletonize(
            all_labels=segmentation,
            teasar_params=self.teasar_kwargs,
            anisotropy=self.spacing,
            object_ids=[self.label],
            **self.non_teasar_kwargs
        )
        try:
            skel = skeleton[self.label]
            vertices = skel.vertices / self.spacing
            edges = skel.edges
            self.skeleton = vertices, edges
        except KeyError:
            logger.warning(f"No centerline found for label {self.label}")
            

    def _postprocess_skeleton(self, **kwargs) -> None:
        """Postprocess the skeleton with ordering and smoothing
        Parameters:
        -----------
        **kwargs:
            min_points: int
                The minimum number of points to keep in the centerline. Default = 25
            max_points: int
                The maximum number of points to keep in the centerline. Default = 200
            smoothing_factor: float
                B-spline smoothing factor (0-1). default 0.5
            gaussian_sigma: float
                The sigma value for the gaussian smoothing. Default = 1.0
        """
        min_points = kwargs.get("min_points", 25)
        max_points = kwargs.get("max_points", 200)
        smoothing_factor = kwargs.get("smoothing_factor", 0.5)
        gaussian_sigma = kwargs.get("gaussian_sigma", 1.0)
        # Order the skeleton
        self._order_skeleton()
        # Resample the centerline with a B-spline
        self._resample_centerline_with_bspline(min_points, max_points, smoothing_factor)
        # Smooth the centerline using a gaussian filter
        self._smooth_centerline(gaussian_sigma)
        # Caclulate the centerline metrics
        segment_lengths, cumulative_lengths, total_length = self._calculate_centerline_metrics()
        self.segment_lengths = segment_lengths
        self.cumulative_lengths = cumulative_lengths
        self.total_length = total_length
        self.succeeded = True

    def _order_skeleton(self) -> None:
        """Order the skeleton"""
        vertices, edges = self.skeleton
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
        # set the centerline
        self.centerline = np.asarray(ordered_vertices)

    def _resample_centerline_with_bspline(self, min_points: int, max_points: int, smoothing_factor: float) -> None:
        """Resample the centerline with a B-spline
        Parameters:
        -----------
        min_points : int
            The minimum number of points to keep in the centerline
        max_points : int
            The maximum number of points to keep in the centerline
        smoothing_factor : float
            The B-spline smoothing factor (0-1)
        """
        centerline = self.centerline
        _, cumulative_lengths, total_length = self._calculate_centerline_metrics()
        # Determine the optimal number of points based on path length
        num_points: int = int(np.clip(int(total_length), min_points, max_points))
        # Fit b-split with appropriate smoothing
        u = np.concatenate([[0], cumulative_lengths])
        u /= u[-1]
        tck, _ = splprep(centerline.T, u=u, s=smoothing_factor, k=3)
        # Generate evenly spaced points along the spline
        u_new = np.linspace(0, 1, num_points)
        new_centerline = np.column_stack(splev(u_new, tck))
        self.centerline = new_centerline

    def _calculate_centerline_metrics(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the centerline segment lengths, cumulative lengths, and total length in spatial units.
        Will be called automatically, but here for accessibility.
        Returns:
        --------
        tuple[np.ndarray, np.ndarray, float]
            The segment lengths, cumulative lengths, and total length
        """
        centerline = self.centerline
        diffs = np.diff(centerline, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        total_length = cumulative_lengths[-1]
        return segment_lengths, cumulative_lengths, total_length

    def _smooth_centerline(self, sigma: float) -> None:
        """Smooth the centerline with a gaussian filter
        Parameters:
        -----------
        sigma : float
            The sigma value for the gaussian filter
        """
        centerline = self.centerline
        # Reflect points at boundaries to avoid edge effects
        n_reflect = int(4*sigma)
        start_reflect = centerline[n_reflect:0:-1]
        end_reflect = centerline[-2:-n_reflect-2:-1]
        # Add padding so that ends are preserved
        extended_centerline = np.vstack([start_reflect, centerline, end_reflect])
        smoothed_centerline = np.zeros_like(centerline)
        # Smooth in each dimension
        for dim in range(3):
            smoothed_centerline[:, dim] = gaussian_filter1d(extended_centerline[:, dim], sigma)
        # Remove padding
        smoothed_centerline = smoothed_centerline[n_reflect:-n_reflect]
        # Preserve the endpoints
        smoothed_centerline[0] = centerline[0]
        smoothed_centerline[-1] = centerline[-1]
        self.centerline = smoothed_centerline




        
