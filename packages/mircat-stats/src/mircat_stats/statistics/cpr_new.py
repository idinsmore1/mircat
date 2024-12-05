import numpy as np

from loguru import logger
from scipy.interpolate import interpn
from mircat_stats.statistics.centerline_new import Centerline

class StraightenedCPR:

    def __init__(self, img: np.ndarray, centerline: Centerline, cross_section_dim: tuple, resolution: int, sigma: int, is_binary: bool):
        """
        Initialize the straightened CPR object
        Parameters
        ----------
        img : np.ndarray
            The image to create the straightened CPR from
        centerline : Centerline
            The centerline object
        cross_section_dim : tuple
            The dimensions of the cross section
        resolution : int
            The resolution of the cross section
        sigma : int
            The sigma for the gaussian filter
        is_binary : bool
            Whether the image is binary
        """
        self.img = img
        self.centerline = centerline
        self.cross_section_dim = cross_section_dim
        self.resolution = resolution
        self.sigma = sigma
        self.is_binary = is_binary
    
    def straighten():
        """
        Straighten the image
        """
        pass



