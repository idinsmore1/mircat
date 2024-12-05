import numpy as np

from loguru import logger
from kimimaro import skeletonize
from scipy.interpolate import interp1d


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
        self.skeleton = None
        self._set_centerline_kwargs(kwargs)


        