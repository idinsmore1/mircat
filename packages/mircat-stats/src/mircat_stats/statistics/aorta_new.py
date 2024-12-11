import numpy as np
import SimpleITK as sitk

from loguru import logger
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.centerline_new import Centerline, calculate_tortuosity
from mircat_stats.statistics.cpr_new import StraightenedCPR
from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.segmentation import Segmentation, SegNotFoundError, ArchNotFoundError
from mircat_stats.statistics.utils import _get_regions


@timer
def calculate_aorta_stats(nifti: MircatNifti) -> dict[str, float]:
    """Calculate the statistics for the aorta in the segmentation.
    Parameters:
    -----------
    nifti : MircatNifti
        The nifti obbject to calculate statistics for
    Returns:
    --------
    dict[str, float]
        The statistics for the aorta
    """
    # Filter to the segmentations we need
    try:
        aorta = Aorta(nifti)
    except SegNotFoundError as e:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return {}
    except Exception as e:
        logger.opt(exception=True).error(f"Error filtering to aorta in {nifti.path}")
        return {}
    # Calculate the aorta statistics
    aorta_stats = aorta.measure_statistics()
    return aorta_stats


class Aorta(Segmentation):
    # This is the list of all vertebrae that could potentially show in specific regions of the aorta
    vertebral_regions_map: dict = {
        "abdominal": [*[f"L{i}" for i in range(1, 6)], "T12L1"],
        "thoracic": [f"T{i}" for i in range(3, 13)],
        "descending": [f"T{i}" for i in range(5, 13)],
    }
    # These are the default values for the aorta
    anisotropic_spacing_mm: tuple = (1, 1, 1)
    cross_section_spacing_mm: tuple = (1, 1)
    cross_section_size_mm: tuple = (100, 100)
    cross_section_resolution: float = 1.0
    root_length_mm: int = 20

    def __init__(self, nifti: MircatNifti):
        super().__init__(nifti, ["aorta", "brachiocephalic_trunk", "subclavian_artery_left"])
        self._check_aortic_regions_in_segmentation()
        self._make_aorta_superior_numpy_array()
        self._get_region_endpoints()

    #### INITIALIZATION OPERATIONS 
    def _check_aortic_regions_in_segmentation(self) -> dict[str, bool]:
        """Check if the aortic regions are present in the segmentation.
        Returns:
        --------
        dict[str, bool]
            Dictionary with keys as region names and values as boolean indicating if the region is present
        """
        vert_midlines = self.vert_midlines
        # L3 has to be in the image for abdominal to be measured
        abdominal = bool(vert_midlines.get("vertebrae_L3_midline", False))
        # T4 and at least T8 need to be in the image for thoracic to be measured
        thoracic = bool(vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get("vertebrae_T4_midline", False))
        # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
        descending = bool(vert_midlines.get("vertebrae_T12_midline", False) and vert_midlines.get("vertebrae_T9_midline", False))
        region_existence = {
            "abdominal": {
                'exists': abdominal, 
            },
            "thoracic": {
                'exists': thoracic, 
            },
            "descending": {
                'exists': descending, 
            }
        }
        self.region_existence = region_existence
        if not any([region_existence[region]['exists'] for region in region_existence]):
            raise SegNotFoundError(f"No aortic regions found in {self.path}")
        return self

    def _make_aorta_superior_numpy_array(self) -> None:
        """Convert the aorta segmentation to a numpy array with the arch at the top and adjust vertebral midlines"""
        self.segmentation_arr = np.flip(np.flipud(sitk.GetArrayFromImage(self.segmentation)), axis=1)
        self.original_ct_arr = np.flip(np.flipud(sitk.GetArrayFromImage(self.original_ct)), axis=1)
        # Adjust the vertebral midlines to account for the flip
        new_midlines = {k: (self.segmentation_arr.shape[0] - 1) - v for k, v in self.vert_midlines.items()}
        self.vert_midlines = new_midlines

    def _get_region_endpoints(self) -> None:
        for region, has_region in self.region_existence.items():
            if has_region:
                endpoints = self._find_aortic_region_endpoints(region, self.vert_midlines)
                self.region_existence[region]['endpoints'] = endpoints
    
    @staticmethod
    def _find_aortic_region_endpoints(region: str, vert_midlines: dict) -> tuple[int, int]:
        possible_locs = Aorta.vertebral_regions_map[region]
        midlines = [
            vert_midlines.get(f"vertebrae_{vert}_midline")
            for vert in possible_locs
            if vert_midlines.get(f"vertebrae_{vert}_midline") is not None
        ]
        midlines = [midline for midline in midlines if midline]
        start = min(midlines)
        end = max(midlines) + 1  # add one to make it inclusive
        return start, end

    #### STATISTICS OPERATIONS
    def measure_statistics(self) -> dict[str, float]:
        """Measure the statistics for the aorta in the segmentation.
        Returns:
        --------
        dict[str, float]
            The statistics for the aorta
        """
        # Create the aorta centerline
        self.setup_stats()
        self._measure_aorta()
        return self.aorta_stats

    def setup_stats(self):
        'Set up the aorta centerline and cprs for statistics'
        (
            self
            ._create_centerline()
            ._create_cpr()
            ._split_main_regions()
        )
        if self.region_existence['thoracic']['exists']:
            self._split_thoracic_regions()
        else:
            self.thoracic_regions = {}

    def _create_centerline(self):
        'Create the centerline for the aorta'
        self.centerline = Centerline(self.anisotropic_spacing_mm)
        abdominal = self.region_existence["abdominal"]['exists']
        thoracic = self.region_existence["thoracic"]['exists']
        descending = self.region_existence["descending"]['exists']
        max_points = 0
        window_length = 10 # mm distance for smoothing
        if abdominal:
            max_points += 300
        # only use either all thoracic or descending
        if thoracic:
            max_points += 400
        elif descending:
            max_points += 200
        self.centerline.create_centerline(self.segmentation_arr, max_points=max_points, window_length=window_length)
        return self

    def _create_cpr(self):
        'Create the CPR for the aorta'
        self.seg_cpr = StraightenedCPR(
            self.segmentation_arr, 
            self.centerline, 
            self.cross_section_size_mm,
            self.cross_section_resolution,
            sigma=2,
            is_binary=True
        ).straighten()
        # self.original_cpr = StraightenedCPR(
        #     self.original_ct_arr, 
        #     self.centerline, 
        #     self.cross_section_size_mm,
        #     self.cross_section_resolution,
        #     sigma=2,
        #     is_binary=False
        # ).straighten()
        return self

    def _split_main_regions(self):
        'Split the centerline and CPR into main aortic regions of abdominal, thoracic and descending'
        regions = {}
        # Split the centerline and cprs into the appropriate_regions
        for region in self.region_existence:
            if self.region_existence[region]['exists']:
                if region == 'descending' and self.region_existence['thoracic']['exists']:
                    continue
                start, end = self.region_existence[region]['endpoints']
                self.region_existence[region]['indices'] = self._split_region(start, end)
        return self

    def _split_region(self, start: int, end: int):
        'Split the centerline and CPR into a specific region'
        valid_indices = []
        for i, point in enumerate(self.centerline.centerline):
            if point[0] >= start and point[0] <= end:
                valid_indices.append(i)
        return valid_indices
    
    def _split_thoracic_regions(self):
        "Split the thoracic aorta centerline and CPRs into ascending, arch, and descending"
        thoracic_regions = {}
        thoracic_indices = self.region_existence['thoracic']['indices']
        thoracic_cpr = self.seg_cpr.cpr_arr[thoracic_indices]
        thoracic_centerline = self.centerline.centerline[thoracic_indices]
        thoracic_cumulative_lengths = self.centerline.cumulative_lengths[thoracic_indices]
        # check if brachiocephalic trunk and left subclavian artery segmentations are present
        arch_segs_in_cpr = np.all(np.isin([2, 3], np.unique(thoracic_cpr)))
        # Split the arch from the ascending and descending
        if arch_segs_in_cpr:
            logger.debug("Using segmentations to define the aortic arch")
            # use the segmentations to define the physical region of the arch
            brach_label = 2
            left_subclavian_label = 3
            # Have to do it this way because we need the start and end based on the
            # thoracic indices, so we can slice with the index
            for slice_idx, cross_section in enumerate(thoracic_cpr):
                if brach_label in cross_section:
                    arch_start = slice_idx
                    break
            for slice_idx, cross_section in enumerate(thoracic_cpr[::-1]):
                if left_subclavian_label in cross_section:
                    arch_end = len(thoracic_cpr) - slice_idx
                    break
        else:
            logger.debug("Using centerline to define the aortic arch")
            # use the top-down view of the aorta to find the arch - less good
            min_pixel_area = 50
            # This is the peak of the centerline
            split = int(self.centerline.centerline[:, 0].min())
            for slice_idx, axial_image in enumerate(self.segmentation_arr):
                regions = _get_regions(axial_image)
                if len(regions) == 2 and slice_idx > split:
                    reg0 = regions[0]
                    reg1 = regions[1]
                    # If both sections of the aorta are sufficiently large,
                    if reg0.area > min_pixel_area and reg1.area > min_pixel_area:
                        split = slice_idx
                        break
            if split is None:
                logger.error("Could not define the aortic arch")
                raise ArchNotFoundError("Could not define the aortic arch")
            for i, point in enumerate(thoracic_centerline):
                if point[0] <= split:
                    arch_start = i
                    break
            for i, point in enumerate(thoracic_centerline[::-1]):
                if point[0] <= split:
                    arch_end = len(thoracic_centerline) - i
                    break
        # Remove the aortic root from the ascending aorta by looking at cumulative length
        asc_start = 0
        for i, length in enumerate(thoracic_cumulative_lengths):
            if length > self.root_length_mm:
                asc_start = i
                break
        thoracic_regions['asc_w_root'] = thoracic_indices[:arch_start]
        thoracic_regions['asc_aorta'] = thoracic_indices[asc_start:arch_start]
        thoracic_regions['aortic_arch'] = thoracic_indices[arch_start:arch_end]
        thoracic_regions['desc_aorta'] = thoracic_indices[arch_end:]
        self.thoracic_regions = thoracic_regions
        return self
        
    def _measure_aorta(self) -> dict[str, float]:
        """Measure the statistics for each region of the aorta. 
        These include maximum diameter, maximum area, length, calcification and periaortic fat.
        Returns:
        --------
        dict[str, float]
            The statistics for the aorta regions
        """
        aorta_stats = {}
        # Get the total aortic stats first
        total_stats = self._measure_region('aorta', [i for i in range(len(self.centerline.centerline))])
        aorta_stats.update(total_stats)
        if self.thoracic_regions:
            for region in self.thoracic_regions:
                if region == 'asc_w_root':
                    continue
                indices = self.thoracic_regions[region]
                aorta_stats.update(self._measure_region(region, indices))
        elif self.region_existence['descending']['exists']:
            aorta_stats.update(self._measure_region('', self.region_existence['descending']['indices']))
        
        if self.region_existence['abdominal']['exists']:
            aorta_stats.update(self._measure_region('abd_aorta', self.region_existence['abdominal']['indices']))
        # Set the aorta stats
        self.aorta_stats = aorta_stats
        return self

    def _measure_region(self, region: str, indices: list[int]) -> dict[str, float]:
        'Measure the statistics for a specific region of the aorta'
        region_stats = {}
        region_centerline = self.centerline.centerline[indices]
        region_cumulative_lengths = self.centerline.cumulative_lengths[indices]
        region_cumulative_lengths = region_cumulative_lengths - region_cumulative_lengths[0]
        region_cpr = (self.seg_cpr.cpr_arr[indices] == 1).astype(np.uint8)
        if hasattr(self, 'original_cpr'):
            region_original_cpr = self.original_cpr.cpr_arr[indices]
        # Region Length
        region_length = round(region_cumulative_lengths[-1], 0)
        region_stats[f'{region}_length_mm'] = region_length
        # Region tortuosity
        region_tortuosity = calculate_tortuosity(region_centerline)
        region_stats.update({f'{region}_{k}': v for k, v in region_tortuosity.items()})
        # Diameters and areas
        if region == 'aorta':
            return region_stats
        
        diameters, max_idx = self._measure_diameters(region_cpr)
        if max_idx is not None:
            max_distance = round(region_cumulative_lengths[max_idx], 0)
            rel_distance = round((max_distance / region_length) * 100, 1)
            diameters['max_diam_from_start_mm'] = max_distance
            diameters['max_diam_rel_distance'] = rel_distance
        region_stats.update({f'{region}_{k}': v for k, v in diameters.items()})
        return region_stats
    
    def _measure_diameters(self, cpr: np.ndarray) -> tuple[dict[str, float], int]:
        '''Measure the maximum, proximal, mid, and distal diameters of the aortic region
        Parameters
        ----------
        cpr: np.ndarray
            The CPR array for the region
        Returns
        -------
        dict[str, float]
            The maximum, proximal, mid, and distal diameters of the aortic region
        int
            the index of the maximum diameter of the CPR
        '''
        out_keys = ['max_diam']
        mid_idx = len(cpr) // 2
        # measure the proximal aortic diameter
        proximal = StraightenedCPR.measure_cross_sectional_diameter(cpr[0], self.cross_section_spacing_mm, diff_threshold=5)
        proximal = {k.replace("max_", "prox_"): proximal[k] for k in out_keys}
        # measure the mid aortic diameter
        mid = StraightenedCPR.measure_cross_sectional_diameter(cpr[mid_idx], self.cross_section_spacing_mm, diff_threshold=5)
        mid = {k.replace("max_", "mid_"): mid[k] for k in out_keys}
        # measure the distal aortic diameter
        distal = StraightenedCPR.measure_cross_sectional_diameter(cpr[-1], self.cross_section_spacing_mm, diff_threshold=5)
        distal = {k.replace("max_", "dist_"): distal[k] for k in out_keys}
        # measure the maximum aortic diameter
        max_diams = []
        max_areas = []
        major_diams = []
        minor_diams = []
        for cross_section in cpr:
            diam = StraightenedCPR.measure_cross_sectional_diameter(cross_section, self.cross_section_spacing_mm, diff_threshold=5)
            max_areas.append(diam['max_area'])
            max_diams.append(diam['max_diam'])
            major_diams.append(diam['major_diam'])
            minor_diams.append(diam['minor_diam'])
        if max_diams:
            largest_idx = np.argmax(max_diams)
            max_ = {
                'max_area': max_areas[largest_idx],
                'avg_diam': max_diams[largest_idx],
                'major_diam': major_diams[largest_idx],
                'minor_diam': minor_diams[largest_idx]
            }
        else:
            max_ = {}
            largest_idx = None
        return {**max_, **proximal, **mid, **distal}, largest_idx