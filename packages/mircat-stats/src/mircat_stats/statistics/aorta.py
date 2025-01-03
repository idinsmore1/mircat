import numpy as np
import SimpleITK as sitk
import pandas as pd

from operator import itemgetter
from loguru import logger
from skimage import draw
from scipy.ndimage import label
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.centerline import Centerline, calculate_tortuosity
from mircat_stats.statistics.cpr import StraightenedCPR
from mircat_stats.statistics.nifti import MircatNifti
from mircat_stats.statistics.segmentation import Segmentation, SegNotFoundError, ArchNotFoundError
from mircat_stats.statistics.utils import _get_regions


@timer
def calculate_aorta_stats(nifti: MircatNifti, contrast: bool) -> dict[str, float]:
    """Calculate the statistics for the aorta in the segmentation.
    Parameters:
    -----------
    nifti : MircatNifti
        The nifti object to calculate statistics for
    contrast : bool
        Whether the image is contrast-enhanced
    Returns:
    --------
    dict[str, float]
        The statistics for the aorta
    """
    # Filter to the segmentations we need
    try:
        aorta = Aorta(nifti, contrast)
        # Calculate the aorta statistics
        aorta_stats = aorta.measure_statistics()
        aorta.write_aorta_stats()
        return aorta_stats
    except SegNotFoundError:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return {}
    except Exception as e:
        logger.opt(exception=True).error(f"Error filtering to aorta in {nifti.path}: {e}")
        return {}
    

class Aorta(Segmentation):
    # This is the list of all vertebrae that could potentially show in specific regions of the aorta
    vertebral_regions_map: dict = {
        "abdominal": ['T12', *[f"L{i}" for i in range(1, 6)]],
        "upper_abd": ['T12', 'L1', 'L2'],
        "lower_abd": ['L2', 'L3', 'L4', 'L5', 'S1'],
        "thoracic": [f"T{i}" for i in range(3, 13)],
        "descending": [f"T{i}" for i in range(5, 13)],
    }
    # These are the default values for the aorta
    anisotropic_spacing_mm: tuple = (1, 1, 1)
    cross_section_spacing_mm: tuple = (1, 1)
    cross_section_size_mm: tuple = (100, 100)
    cross_section_resolution: float = 1.0
    root_length_mm: int = 20

    def __init__(self, nifti: MircatNifti, contrast: bool):
        super().__init__(nifti, ["aorta", "brachiocephalic_trunk", "subclavian_artery_left"])
        self.contrast_enhanced = contrast
        self._make_SPR_numpy_array()

    #### INITIALIZATION OPERATIONS
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
        end = max(midlines)  # add one to make it inclusive
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
        try:
            self.setup_stats()
            self.calculate_stats()
            return self.aorta_stats
        except ArchNotFoundError:
            logger.error(f"Could not define aortic arch in {self.path}")
            return {}

    def setup_stats(self):
        "Set up the aorta centerline and cprs for statistics"
        self._create_centerline()._create_cpr()._split_aorta_regions()

    def _create_centerline(self):
        "Create the centerline for the aorta"
        self.centerline = Centerline(self.anisotropic_spacing_mm)
        abdominal = bool(self.vert_midlines.get('vertebrae_L3_midline', False))
        thoracic = bool(self.vert_midlines.get('vertebrae_T4_midline', False) and self.vert_midlines.get('vertebrae_T8_midline', False))
        descending = bool(self.vert_midlines.get('vertebrae_T12_midline', False) and self.vert_midlines.get('vertebrae_T9_midline', False))
        max_points = 0
        window_length = 10  # mm distance for smoothing
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
        "Create the CPR for the aorta"
        self.seg_cpr = StraightenedCPR(
            self.segmentation_arr,
            self.centerline,
            self.cross_section_size_mm,
            self.cross_section_resolution,
            sigma=2,
            is_binary=True,
        ).straighten()
        self.original_cpr = StraightenedCPR(
            self.original_ct_arr,
            self.centerline,
            self.cross_section_size_mm,
            self.cross_section_resolution,
            sigma=2,
            is_binary=False
        ).straighten()
        return self

    def _split_aorta_regions(self):
        "Split the centerline and CPR into aortic regions of root, ascending, arch, descending, upper abdominal and lower abdominal"
        # Split the centerline and cprs into the appropriate_regions
        aorta_regions = {}
        # Need at least T4 and T8 to capture enough of the thoracic aorta to be useful
        thoracic = bool(self.vert_midlines.get("vertebrae_T4_midline") is not None and self.vert_midlines.get("vertebrae_T8_midline") is not None)
        # Need at least the T9 and T12 to capture enough of the descending aorta to be useful
        descending = bool(self.vert_midlines.get("vertebrae_T9_midline") is not None and self.vert_midlines.get("vertebrae_T12_midline") is not None)
        if thoracic:
            start, end = self._find_aortic_region_endpoints("thoracic", self.vert_midlines)
            indices = self._get_region_indices(start, end)
            thor_regions = self._split_thoracic_regions(indices)
            aorta_regions.update(thor_regions)
        elif descending:  # Only need to find descending if the full thoracic aorta is not present
            start, end = self._find_aortic_region_endpoints("descending", self.vert_midlines)
            indices = self._get_region_indices(start, end)
            aorta_regions["desc_aorta"] = indices
        # Upper abdominal aorta is between T12 and L2, but L1 will suffice for existence as it may be cut off
        upper_abd = bool(self.vert_midlines.get('vertebrae_T12_midline') is not None and self.vert_midlines.get('vertebrae_L1_midline') is not None)
        # Lower abdominal just needs L2 and L3- will check below when getting end points
        lower_abd = bool(self.vert_midlines.get('vertebrae_L2_midline') is not None and self.vert_midlines.get('vertebrae_L3_midline') is not None)
        if upper_abd:
            start, end = self._find_aortic_region_endpoints("upper_abd", self.vert_midlines)
            indices = self._get_region_indices(start, end)
            aorta_regions["up_abd_aorta"] = indices
        if lower_abd:
            start, end = self._find_aortic_region_endpoints("lower_abd", self.vert_midlines)
            indices = self._get_region_indices(start, end)
            aorta_regions["lw_abd_aorta"] = indices
        self.aorta_regions = aorta_regions
        return self

    def _get_region_indices(self, start: int, end: int):
        "Split the centerline and CPR into a specific region"
        valid_indices = []
        for i, point in enumerate(self.centerline.coordinates):
            if point[0] >= start and point[0] <= end:
                valid_indices.append(i)
        return valid_indices

    def _split_thoracic_regions(self, indices: list[int]) -> dict[str, list[int]]:
        """Split the thoracic aorta centerline and CPRs into root, ascending, arch, and descending
        Parameters
        ----------
        indices: list[int]
            The indices of the thoracic aorta for the centerline and CPR
        Returns
        -------
        dict[str, list[int]]
            The dictionary of indices for the thoracic aorta regions
        """
        thoracic_regions = {}
        # thoracic_indices = self.region_existence["thoracic"]["indices"]
        thoracic_indices = indices.copy()
        thoracic_cpr = self.seg_cpr.array[thoracic_indices]
        thoracic_centerline = self.centerline.coordinates[thoracic_indices]
        thoracic_cumulative_lengths = self.centerline.cumulative_lengths[thoracic_indices]
        # check if brachiocephalic trunk and left subclavian artery segmentations are present
        arch_segs_in_cpr = np.all(np.isin([2, 3], np.unique(thoracic_cpr)))
        # Split the arch from the ascending and descending
        if arch_segs_in_cpr:
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
            # use the top-down view of the aorta to find the arch - less good
            min_pixel_area = 50
            # This is the peak of the centerline
            split = int(self.centerline.coordinates[:, 0].min())
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
            arch_start = None
            arch_end = None
            for i, point in enumerate(thoracic_centerline):
                if point[0] <= split:
                    arch_start = i
                    break
            for i, point in enumerate(thoracic_centerline[::-1]):
                if point[0] <= split:
                    arch_end = len(thoracic_centerline) - i
                    break
            if arch_start is None or arch_end is None:
                logger.error("Could not define the aortic arch")
                raise ArchNotFoundError("Could not define the aortic arch")
        # Remove the aortic root from the ascending aorta by looking at cumulative length
        asc_start = 0
        for i, length in enumerate(thoracic_cumulative_lengths):
            if length > self.root_length_mm:
                asc_start = i
                break
        thoracic_regions["aortic_root"] = thoracic_indices[:asc_start]
        thoracic_regions["asc_aorta"] = thoracic_indices[asc_start:arch_start]
        thoracic_regions["aortic_arch"] = thoracic_indices[arch_start:arch_end]
        thoracic_regions["desc_aorta"] = thoracic_indices[arch_end:]
        return thoracic_regions

    def calculate_stats(self) -> dict[str, float]:
        """Calculate the statistics for each region of the aorta.
        These include maximum diameter, maximum area, length, calcification and periaortic fat.
        Returns:
        --------
        dict[str, float]
            The statistics for the aorta regions
        """
        # Get the total aortic stats first
        aorta_stats = self._measure_whole_aorta()
        # Measure the aorta regions   
        for region, indices in self.aorta_regions.items():
            aorta_stats.update(self._measure_region(region, indices))
        # Set the aorta stats
        self.aorta_stats = aorta_stats
        return self
    
    def _measure_whole_aorta(self) -> dict[str, float]:
        """Measure the statistics for the whole aorta.
        Sets the following attributes after measurement:
            aorta_diameters: list[dict[str, float]] -> a list of measurement dictionaries for each cross section
            aorta_fat: dict[str, float] -> the output dictionary for the fat measurements
        Returns:
        --------
        dict[str, float]
            The statistics for the whole aorta
        """
        aorta_stats = {}
        # Set the total aorta length
        cumulative_length = self.centerline.cumulative_lengths[-1]
        aorta_stats['aorta_length_mm'] = round(cumulative_length, 0)
        # Calculate tortuosity
        centerline = self.centerline.coordinates
        tortuosity = calculate_tortuosity(centerline)
        tortuosity = {f"aorta_{k}": v for k, v in tortuosity.items()}
        aorta_stats.update(tortuosity)
        # Measure diameters for each slice of the cpr
        self._measure_all_cross_sections()
        # Measure the periaortic fat
        periaortic_stats = self._measure_total_periaortic_fat()
        aorta_stats.update(periaortic_stats)
        # Measure the total aortic calcification
        calcification_stats = self._measure_total_aortic_calcification()
        aorta_stats.update(calcification_stats)
        return aorta_stats

    def _measure_all_cross_sections(self):
        seg_cpr = self.seg_cpr.array
        cross_section_data = []
        for cross_section in seg_cpr:
            cross_section_measures = StraightenedCPR.measure_cross_section(
                cross_section, self.cross_section_spacing_mm, diff_threshold=5
            )
            cross_section_data.append(cross_section_measures)
        self.cross_section_data = cross_section_data
        return
    
    def _measure_total_periaortic_fat(self) -> dict[str, float]:
        """Measure the periaortic fat for the aorta
        Returns
        -------
        dict[str, float]
            The periaortic fat statistics
        """
        periaortic_fat_stats = {}
        seg_cpr = self.seg_cpr.array
        cross_section_data = self.cross_section_data
        self._create_periaortic_arrays(cross_section_data)
        periaortic_fat_stats['aorta_periaortic_total_cm3'] = round((self.periaortic_mask_cpr.sum() + seg_cpr.sum()) / 1000, 1)
        periaortic_fat_stats['aorta_periaortic_ring_cm3'] = round(self.periaortic_mask_cpr.sum() / 1000, 1)
        periaortic_fat_stats['aorta_periaortic_fat_cm3'] = round(self.periaortic_fat_cpr.sum() / 1000, 1)
        fat_values = np.where(self.periaortic_fat_cpr == 1, self.original_cpr.array, np.nan)
        periaortic_fat_stats['aorta_periaortic_fat_mean_hu'] = round(np.nanmean(fat_values), 1)
        periaortic_fat_stats['aorta_periaortic_fat_stddev_hu'] = round(np.nanstd(fat_values), 1)
        return periaortic_fat_stats 
    
    def _create_periaortic_arrays(self, aortic_diameters: list[dict[str, float]]) -> np.ndarray:
        """Create the periaortic fat array for the aorta
        Parameters
        ----------
        aortic_diameters: list[dict[str, float]]
            The list of aortic diameters for each cross section
        Returns
        -------
        np.ndarray
            The array of masked periaortic fat
        """
        diams = [d.get('diam', np.nan) for d in aortic_diameters]
        seg_cpr = self.seg_cpr.array
        ct_cpr = np.clip(self.original_cpr.array, -250, 250)  # clip to HU range to remove artifacts
        periaortic_fat = np.zeros_like(seg_cpr, dtype=np.uint8)
        periaortic_mask = np.zeros_like(seg_cpr, dtype=np.uint8)
        assert len(diams) == len(seg_cpr), ValueError("Number of diameters and CPR slices must match")
        for i, (diam, cpr_slice, ct_slice) in enumerate(zip(diams, seg_cpr, ct_cpr)):
            if np.isnan(diam) or diam == 0:
                continue
            radius = (diam / 2) + 10 # add 10mm to the radius
            # Draw a filled circle around the center of the aorta
            center_y, center_x = _get_regions(cpr_slice)[0].centroid
            ring_mask = np.zeros_like(cpr_slice, dtype=np.uint8)
            rr, cc = draw.disk((center_y, center_x), radius, shape=cpr_slice.shape)
            ring_mask[rr, cc] = 1
            # Remove the aorta from the mask
            ring_mask[cpr_slice == 1] = 0
            periaortic_mask[i] = ring_mask
            # Remove any non-fat regions inside the ring
            fat_mask = (ct_slice >= -190) & (ct_slice <= -30) * ring_mask
            periaortic_fat[i] = fat_mask
        self.periaortic_mask_cpr = periaortic_mask
        self.periaortic_fat_cpr = periaortic_fat

    def _measure_total_aortic_calcification(self) -> dict[str, float]:
        """Measure the calcification for the entire aorta found in the image
        Returns
        -------
        dict[str, float]
            The calcification statistics for the aorta
        """
        self._create_calcification_mask()
        self._mark_calcifications()
        # 
        total_calcification = self.calcification_mask.sum()  # mm3 volume of calcification
        if total_calcification > 0:
            weighted_calc_score = total_calcification * np.mean(self.original_cpr.array[self.calcification_mask == 1]) # weighted by HU
        else:
            weighted_calc_score = None
        calcification_stats = {}
        calcification_stats['aorta_agatston_calcs'] = self.cpr_calcs.sum()
        calcification_stats['aorta_agatston_score'] = self.cpr_agatstons.sum()
        calcification_stats['aorta_calcification_mm3'] = total_calcification
        calcification_stats['aorta_weighted_calc_score'] = weighted_calc_score
        return calcification_stats

    def _mark_calcifications(self) -> None:
        """Track the calcifications in the aorta"""
        slice_calcifications = []
        slice_areas = []
        slice_agatstons = []
        min_volume = 1 # mm^3
        for ct_slice, calc_slice in zip(self.original_cpr.array, self.calcification_mask):
            labeled_array, _ = label(calc_slice)
            unique_labels, areas = np.unique(labeled_array[labeled_array != 0], return_counts=True)
            num_calcifications = np.sum(areas >= min_volume)
            slice_agatston = 0
            slice_area = 0
            for label_, area in zip(unique_labels, areas):
                if area >= min_volume:
                    avg_hu = np.mean(ct_slice[labeled_array == label_])
                    if 130 <= avg_hu < 200:
                        factor = 1
                    elif 200 <= avg_hu < 300:
                        factor = 2
                    elif 300 <= avg_hu < 400:
                        factor = 3
                    elif avg_hu >= 400:
                        factor = 4
                    else:
                        factor = 0
                    slice_agatston += area * factor
                    slice_area += area
            slice_calcifications.append(num_calcifications)
            slice_areas.append(slice_area)
            slice_agatstons.append(slice_agatston)
        self.cpr_calcs = np.array(slice_calcifications, dtype=np.uint16)
        self.cpr_calc_areas = np.array(slice_areas, dtype=np.uint64)
        self.cpr_agatstons = np.array(slice_agatstons, dtype=np.uint64)

                    
    def _create_calcification_mask(self) -> np.ndarray:
        """Create the calcification mask for the aorta
        Returns
        -------
        np.ndarray
            The array of masked calcifications
        """
        calcification_hu_threshold = 320 if self.contrast_enhanced else 130
        seg = self.seg_cpr.array
        aorta_only = (seg == 1).astype(np.uint8)  # Limit to only the aorta if it is thoracic
        ct = self.original_cpr.array
        calcification_mask = (ct >= calcification_hu_threshold).astype(np.uint8)
        self.calcification_mask = calcification_mask * aorta_only
        return calcification_mask


    def _measure_region(self, region: str, indices: list[int]) -> dict[str, float]:
        "Measure the statistics for a specific region of the aorta"
        region_stats = {}
        if len(indices) < 3:
            return region_stats # not enough points to measure
        try:
            # Region length
            region_cumulative_lengths = self.centerline.cumulative_lengths[indices]
            region_cumulative_lengths = region_cumulative_lengths - region_cumulative_lengths[0]
            region_length = round(region_cumulative_lengths[-1], 0)
            region_stats[f"{region}_length_mm"] = region_length
            # Region tortuosity
            region_centerline = self.centerline.coordinates[indices]
            region_tortuosity = calculate_tortuosity(region_centerline)
            region_stats.update({f"{region}_{k}": v for k, v in region_tortuosity.items()})
            # Diameters and areas
            region_diameters = list(itemgetter(*indices)(self.cross_section_data))
            diameters, max_idx = self._extract_region_diameters(region_diameters)
            if max_idx is not None:
                max_distance = round(region_cumulative_lengths[max_idx], 0)
                rel_distance = round((max_distance / region_length) * 100, 1)
                diameters["max_diam_dist_mm"] = max_distance  # distance from the start of the region
                diameters["max_diam_rel_dist"] = rel_distance  # relative distance from the start of the region
            region_stats.update({f"{region}_{k}": v for k, v in diameters.items()})
            # Periaortic fat
            fat_measures = self._extract_region_periaortic_fat(region, indices)
            region_stats.update(fat_measures)
            # Calcification
            calcification_stats = self._extract_region_calcification(region, indices)
            region_stats.update(calcification_stats)
        except IndexError:
            logger.error(f"Index Error measuring {region} region in {self.path}")
        finally:
            return region_stats
    
    def _extract_region_diameters(self, region_diameters: list[str, dict]) -> tuple[dict[str, float], int]:
        """Measure the maximum, proximal, mid, and distal diameters of the aortic region
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
        """
        # extract the proximal region diameter
        for i, diam in enumerate(region_diameters):
            if not np.isnan(diam["diam"]):
                prox_idx = i
                break
        proximal = {f'prox_{k}': v for k, v in region_diameters[prox_idx].items()}
        # extract the mid region diameter
        mid_idx = len(region_diameters) // 2
        mid = {f'mid_{k}': v for k, v in region_diameters[mid_idx].items()}
        # extract the distal region diameter
        for i, diam in enumerate(region_diameters[::-1]):
            if not np.isnan(diam.get('diam', np.nan)):
                dist_idx = i
                break
        distal = {f'dist_{k}': v for k, v in region_diameters[::-1][dist_idx].items()}
        # measure the maximum aortic diameter
        diams = []
        areas = []
        major_axes = []
        minor_axes = []
        for diam in region_diameters:
            diams.append(diam.get("diam", np.nan))
            major_axes.append(diam.get("major_axis", np.nan))
            minor_axes.append(diam.get("minor_axis", np.nan))
            areas.append(diam.get("area", np.nan))
        if diams:
            largest_idx = np.nanargmax(diams)
            max_ = {
                "max_diam": diams[largest_idx],
                "max_major_axis": major_axes[largest_idx],
                "max_minor_axis": minor_axes[largest_idx],
                "max_area": areas[largest_idx],
            }
        else:
            max_ = {}
            largest_idx = None
        return {**max_, **proximal, **mid, **distal}, largest_idx 

    def _extract_region_periaortic_fat(self, region, indices) -> dict[str, float]:
        measures = {}
        region_seg_cpr = self.seg_cpr.array[indices] 
        region_ct_cpr = self.original_cpr.array[indices]
        region_mask = self.periaortic_mask_cpr[indices]
        region_fat = self.periaortic_fat_cpr[indices]
        measures[f'{region}_periaortic_total_cm3'] = round((region_mask.sum() + region_seg_cpr.sum()) / 1000, 1)
        measures[f"{region}_periaortic_ring_cm3"] = round(region_mask.sum() / 1000, 1)
        measures[f"{region}_periaortic_fat_cm3"] = round(region_fat.sum() / 1000, 1)
        # Calculate the average intensity and standard deviation of the fat
        fat_values = np.where(region_fat == 1, region_ct_cpr, np.nan)
        measures[f'{region}_periaortic_fat_mean_hu'] = round(np.nanmean(fat_values), 1)
        measures[f'{region}_periaortic_fat_stddev_hu'] = round(np.nanstd(fat_values), 1)
        return measures

    def _extract_region_calcification(self, region, indices) -> dict[str, float]:
        measures = {}
        region_calc_cpr = self.calcification_mask[indices]
        region_ct_cpr = self.original_cpr.array[indices]
        region_calcs = self.cpr_calcs[indices]
        region_agatstons = self.cpr_agatstons[indices]
        measures[f"{region}_agatston_calcs"] = region_calcs.sum()
        measures[f"{region}_agatston_score"] = region_agatstons.sum()
        total_calcification = region_calc_cpr.sum()  # mm3 volume of calcification
        if total_calcification > 0:
            weighted_calc_score = total_calcification * np.mean(region_ct_cpr[region_calc_cpr == 1]) # weighted by HU
        else:
            weighted_calc_score = None
        measures[f"{region}_calcification_mm3"] = total_calcification
        measures[f"{region}_weighted_calc_score"] = weighted_calc_score
        return measures
    
    #### Write the statistics to a csv file
    def write_aorta_stats(self) -> None:
        """Write the aorta statistics to a csv file"""
        index, z, y, x = [], [], [], []
        for i, point in enumerate(self.centerline.coordinates):
            index.append(i)
            z.append(point[0].round(1))
            y.append(point[1].round(1))
            x.append(point[2].round(1))
        regions = [None for _ in  range(len(index))]
        name_map = {
            'aortic_root': 'root',
            'asc_aorta': 'ascending', 
            'aortic_arch': 'arch', 
            'desc_aorta': 'descending', 
            'up_abd_aorta': 'upper_abdominal', 
            'lw_abd_aorta': 'lower_abdominal'
        }
        for region, indices in self.aorta_regions.items():
            for idx in indices:
                regions[idx] = name_map.get(region)
        segment_lengths = [0, *self.centerline.segment_lengths.round(2).tolist()]
        cumulative_lengths = self.centerline.cumulative_lengths.round(2).tolist()
        diameters = [d.get('diam') for d in self.cross_section_data]
        major_axes = [d.get('major_axis') for d in self.cross_section_data]
        minor_axes = [d.get('minor_axis') for d in self.cross_section_data]
        areas = [d.get('area') for d in self.cross_section_data]
        flatnesses = [d.get('flatness') for d in self.cross_section_data]
        roundnesses = [d.get('roundness') for d in self.cross_section_data]
        calcs = self.cpr_calcs.tolist()
        calc_areas = self.cpr_calc_areas.tolist()
        agatstons = self.cpr_agatstons.tolist()
        # total_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[0].tolist()], None, None, None]
        # in_plane_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[1].tolist()], None, None, None]
        # torsional_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[2].tolist()], None, None, None]
        df = pd.DataFrame(
            {
                "centerline_index": index,
                "region": regions,
                "z_coordinate": z,
                "y_coordinate": y,
                "x_coordinate": x,
                "segment_length_mm": segment_lengths,
                "cumulative_length_mm": cumulative_lengths,
                "area": areas,
                "diameter": diameters,
                "major_axis": major_axes,
                "minor_axis": minor_axes,
                "flatness": flatnesses,
                "roundness": roundnesses,
                "calcifications": calcs,
                "calcification_area": calc_areas,
                "agatston_score": agatstons,
            },
        )

        output_path = self.seg_folder / f'{self.nifti_name}_aorta.csv'
        df.to_csv(output_path, index=False)
        logger.log("AORTA", f"Aorta statistics written to {output_path}", extra={'output_path': str(output_path.absolute())})