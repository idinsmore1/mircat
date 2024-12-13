import numpy as np
import SimpleITK as sitk
import polars as pl

from operator import itemgetter
from loguru import logger
from skimage import draw
from mircat_stats.configs.logging import timer
from mircat_stats.statistics.centerline import Centerline, calculate_tortuosity
from mircat_stats.statistics.cpr import StraightenedCPR
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
    except SegNotFoundError:
        logger.opt(exception=True).error(f"No aorta found in {nifti.path}")
        return {}
    except Exception:
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
        thoracic = bool(
            vert_midlines.get("vertebrae_T8_midline", False) and vert_midlines.get("vertebrae_T4_midline", False)
        )
        # If at least the T12 and T9 are in the image, then we can measure the descending separate from the ascending
        descending = bool(
            vert_midlines.get("vertebrae_T12_midline", False) and vert_midlines.get("vertebrae_T9_midline", False)
        )
        region_existence = {
            "abdominal": {
                "exists": abdominal,
            },
            "thoracic": {
                "exists": thoracic,
            },
            "descending": {
                "exists": descending,
            },
        }
        self.region_existence = region_existence
        if not any([region_existence[region]["exists"] for region in region_existence]):
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
        for region_name, region in self.region_existence.items():
            if region["exists"]:
                endpoints = self._find_aortic_region_endpoints(region_name, self.vert_midlines)
                self.region_existence[region_name]["endpoints"] = endpoints

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
        try:
            self.setup_stats()
            self.calc_stats()
            return self.aorta_stats
        except ArchNotFoundError:
            logger.error(f"Could not define aortic arch in {self.path}")
            return {}

    def setup_stats(self):
        "Set up the aorta centerline and cprs for statistics"
        self._create_centerline()._create_cpr()._split_main_regions()
        if self.region_existence["thoracic"]["exists"]:
            self._split_thoracic_regions()
        else:
            self.thoracic_regions = {}

    def _create_centerline(self):
        "Create the centerline for the aorta"
        self.centerline = Centerline(self.anisotropic_spacing_mm)
        abdominal = self.region_existence["abdominal"]["exists"]
        thoracic = self.region_existence["thoracic"]["exists"]
        descending = self.region_existence["descending"]["exists"]
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

    def _split_main_regions(self):
        "Split the centerline and CPR into main aortic regions of abdominal, thoracic and descending"
        # Split the centerline and cprs into the appropriate_regions
        for region in self.region_existence:
            if self.region_existence[region]["exists"]:
                if region == "descending" and self.region_existence["thoracic"]["exists"]:
                    continue
                start, end = self.region_existence[region]["endpoints"]
                self.region_existence[region]["indices"] = self._split_region(start, end)
        return self

    def _split_region(self, start: int, end: int):
        "Split the centerline and CPR into a specific region"
        valid_indices = []
        for i, point in enumerate(self.centerline.centerline):
            if point[0] >= start and point[0] <= end:
                valid_indices.append(i)
        return valid_indices

    def _split_thoracic_regions(self):
        "Split the thoracic aorta centerline and CPRs into ascending, arch, and descending"
        thoracic_regions = {}
        thoracic_indices = self.region_existence["thoracic"]["indices"]
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
        self.thoracic_regions = thoracic_regions
        return self

    def calc_stats(self) -> dict[str, float]:
        """Calculate the statistics for each region of the aorta.
        These include maximum diameter, maximum area, length, calcification and periaortic fat.
        Returns:
        --------
        dict[str, float]
            The statistics for the aorta regions
        """
        # Get the total aortic stats first
        aorta_stats = self._measure_whole_aorta()
        if self.thoracic_regions:
            for region in self.thoracic_regions:
                indices = self.thoracic_regions[region]
                aorta_stats.update(self._measure_region(region, indices))
        elif self.region_existence["descending"]["exists"]:
            aorta_stats.update(self._measure_region("", self.region_existence["descending"]["indices"]))
        if self.region_existence["abdominal"]["exists"]:
            aorta_stats.update(self._measure_region("abd_aorta", self.region_existence["abdominal"]["indices"]))
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
        centerline = self.centerline.centerline
        tortuosity, angle_measures = calculate_tortuosity(centerline)
        tortuosity = {f"aorta_{k}": v for k, v in tortuosity.items()}
        self.angles_of_centerline = angle_measures
        aorta_stats.update(tortuosity)
        # Measure diameters for each slice of the cpr
        seg_cpr = self.seg_cpr.cpr_arr
        aorta_diameters = []
        for cross_section in seg_cpr:
            diam = StraightenedCPR.measure_cross_section(
                cross_section, self.cross_section_spacing_mm, diff_threshold=5
            )
            aorta_diameters.append(diam)
        self.aorta_diameters = aorta_diameters
        # Create the periaortic fat array
        self._create_periaortic_arrays(aorta_diameters)
        # The cpr is always in (1, 1, 1) mm spacing, so the sum will be in mm^3
        aorta_stats['aorta_periaortic_total_cm3'] = round((self.periaortic_mask_cpr.sum() + seg_cpr.sum()) / 1000, 1)
        aorta_stats['aorta_periaortic_ring_cm3'] = round(self.periaortic_mask_cpr.sum() / 1000, 1)
        aorta_stats['aorta_periaortic_fat_cm3'] = round(self.periaortic_fat_cpr.sum() / 1000, 1)
        fat_values = np.where(self.periaortic_fat_cpr == 1, self.original_cpr.cpr_arr, np.nan)
        aorta_stats['aorta_periaortic_fat_mean_hu'] = round(np.nanmean(fat_values), 1)
        aorta_stats['aorta_periaortic_fat_stddev_hu'] = round(np.nanstd(fat_values), 1)
        return aorta_stats
    
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
        diams = [d["diam"] for d in aortic_diameters]
        seg_cpr = self.seg_cpr.cpr_arr
        ct_cpr = np.clip(self.original_cpr.cpr_arr, -250, 250)  # clip to HU range to remove artifacts
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

    def _measure_region(self, region: str, indices: list[int]) -> dict[str, float]:
        "Measure the statistics for a specific region of the aorta"
        region_stats = {}
        try:
            # Region length
            region_cumulative_lengths = self.centerline.cumulative_lengths[indices]
            region_cumulative_lengths = region_cumulative_lengths - region_cumulative_lengths[0]
            region_length = round(region_cumulative_lengths[-1], 0)
            region_stats[f"{region}_length_mm"] = region_length
            # Region tortuosity
            region_centerline = self.centerline.centerline[indices]
            region_tortuosity, _ = calculate_tortuosity(region_centerline)
            region_stats.update({f"{region}_{k}": v for k, v in region_tortuosity.items()})
            # Diameters and areas
            region_diameters = list(itemgetter(*indices)(self.aorta_diameters))
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
        except IndexError:
            logger.error(f"Index Error measuring {region} region in {self.path}")
        finally:
            return region_stats

    def _extract_region_periaortic_fat(self, region, indices):
        measures = {}
        region_seg_cpr = self.seg_cpr.cpr_arr[indices] 
        region_ct_cpr = self.original_cpr.cpr_arr[indices]
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
            if not np.isnan(diam["diam"]):
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
    
    #### Write the statistics to a csv file
    def write_aorta_stats(self) -> None:
        """Write the aorta statistics to a csv file"""
        index, z, y, x = [], [], [], []
        for i, point in enumerate(self.centerline.centerline):
            index.append(i)
            z.append(point[0].round(1))
            y.append(point[1].round(1))
            x.append(point[2].round(1))
        regions = [None for _ in  range(len(index))]
        name_map = {'aortic_root': 'root', 'asc_aorta': 'ascending', 'aortic_arch': 'arch', 'desc_aorta': 'descending', 'abd_aorta': 'abdominal'}
        if self.thoracic_regions:
            for region, indices in self.thoracic_regions.items():
                for idx in indices:
                    regions[idx] = name_map.get(region)
        elif self.region_existence["descending"]["exists"]:
            for i in self.region_existence["descending"]["indices"]:
                regions[i] = "descending"
        if self.region_existence['abdominal']['exists']:
            for i in self.region_existence['abdominal']['indices']:
                regions[i] = 'abdominal'
        segment_lengths = [0, *self.centerline.segment_lengths.round(2).tolist()]
        cumulative_lengths = self.centerline.cumulative_lengths.round(2).tolist()
        diameters = [d['diam'] for d in self.aorta_diameters]
        major_axes = [d['major_axis'] for d in self.aorta_diameters]
        minor_axes = [d['minor_axis'] for d in self.aorta_diameters]
        areas = [d['area'] for d in self.aorta_diameters]
        flatnesses = [d['flatness'] for d in self.aorta_diameters]
        roundnesses = [d['roundness'] for d in self.aorta_diameters]
        total_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[0].tolist()], None, None]
        in_plane_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[1].tolist()], None, None]
        torsional_angles = [0, *[round(x, 2) for x in self.angles_of_centerline[2].tolist()], None, None]
        for angle_list in [total_angles, in_plane_angles, torsional_angles]:
            for i, angle in enumerate(angle_list):
                if angle is not None:
                    angle_val = round(np.rad2deg(angle))
                    if angle_val == 180:
                        angle_val = 0
                    elif angle_val > 90:
                        angle_val = 180 - angle_val
                    angle_list[i] = angle_val
        df = pl.DataFrame(
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
                "total_angle": total_angles,
                "in_plane_angle": in_plane_angles,
                "torsional_angle": torsional_angles,
            },
            strict=False,
            nan_to_null=True,
        )
        output_path = self.seg_folder / f'{self.nifti_name}_aorta.csv'
        df.write_csv(output_path)
