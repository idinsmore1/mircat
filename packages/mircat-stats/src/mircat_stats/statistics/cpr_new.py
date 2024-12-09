import numpy as np

from dataclasses import dataclass
from loguru import logger
from scipy.interpolate import interpn
from mircat_stats.statistics.centerline_new import Centerline
from mircat_stats.statistics.utils import _get_regions
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes

@dataclass
class StraightenedCPR:
    """Class for creating and representing a Straightened Curved Planar Reformation"""
    img: np.ndarray
    centerline: Centerline
    cross_section_dim: tuple
    resolution: float
    sigma: int
    is_binary: bool

    def straighten(self):
        """
        Straighten the image
        """
        cpr = []
        empty_slices = []
        for i in range(len(self.centerline.centerline)):
            cross_section = self.extract_orthogonal_cross_section(i)
            if self.is_binary:
                cross_section = StraightenedCPR._postprocess_cross_section(cross_section, self.sigma)
                if cross_section.sum() == 0:
                    empty_slices.append(1)
                else:
                    empty_slices.append(0)
            cpr.append(cross_section)
            
        cpr = np.stack(cpr, axis=0)
        # remove the first and last cross-sections to avoid odd cuts
        self.cpr_arr = cpr
        self.empty_slices = empty_slices
        return self

    def extract_orthogonal_cross_section(self, index: int):
        """
        Extract an orthogonal cross section from the image
        Parameters
        ----------
        array : np.ndarray
            The image array
        center_point : np.ndarray
            The center point
        tangent_vector : np.ndarray
            The tangent vector
        normals : list[list]
            The normal vectors to the tangent
        resolution : float
        """
        # Set up the points and vectors
        arr = self.img
        center_point = self.centerline.centerline[index]
        v1 = self.centerline.binormal_vectors[0][index]
        v2 = self.centerline.binormal_vectors[1][index]
        if v1 is None or v2 is None:
            raise ValueError("Invalid normal vectors")
        width, height = self.cross_section_dim
        resolution = self.resolution   
        # Create a mesh grid to hold the output
        x_lin = np.linspace(-width / 2, width / 2, int(width / resolution))
        y_lin = np.linspace(-height / 2, height / 2, int(height / resolution))
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        # Map the grid points to the 3D array indices
        slice_points = center_point + x_grid[..., np.newaxis] * v1 + y_grid[..., np.newaxis] * v2
        if self.is_binary:
            slice_points = np.rint(slice_points).astype(int)
            # Initialize an empty slice with zeros (padding)
            slice_2d = np.zeros((int(height / resolution), int(width / resolution)), dtype=arr.dtype)
            if arr.min() != 0:
                slice_2d += arr.min()
            # Compute valid index ranges considering the boundaries
            valid_x = (slice_points[..., 0] >= 0) & (slice_points[..., 0] < arr.shape[0])
            valid_y = (slice_points[..., 1] >= 0) & (slice_points[..., 1] < arr.shape[1])
            valid_z = (slice_points[..., 2] >= 0) & (slice_points[..., 2] < arr.shape[2])
            valid_indices = valid_x & valid_y & valid_z
            # Extract values for valid indices and assign to the slice, leave zeros elsewhere
            valid_points = slice_points[valid_indices]
            slice_2d[valid_indices] = arr[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]]
        else:
            x = np.arange(arr.shape[0])
            y = np.arange(arr.shape[1])
            z = np.arange(arr.shape[2])
            slice_2d = interpn((x, y, z), arr, slice_points, method="linear", bounds_error=False, fill_value=arr.min())
        return slice_2d

    @staticmethod
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
            center_label = StraightenedCPR._filter_label_regions(cross_section, label)
            center_label = remove_small_holes(center_label)
            center_label = gaussian(center_label, sigma=sigma).round(0)
            # Assign the output of the binary to the appropriate label
            output_cross_section[center_label == 1] = label
        return output_cross_section

    @staticmethod
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
            centered_label = StraightenedCPR._closest_to_centroid(tmp_cross_section, regions)
        else:
            centered_label = tmp_cross_section
        return centered_label > 0

    @staticmethod
    def _closest_to_centroid(cross_section: np.ndarray, regions: list) -> np.ndarray:
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