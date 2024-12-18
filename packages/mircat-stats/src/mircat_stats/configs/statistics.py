aorta_region_columns = [
    "length_mm",
    "tort_idx",
    "max_diam",
    "max_major_axis",
    "max_minor_axis",
    "max_diam_dist_mm",
    "max_diam_rel_dist",
    "max_area",
    "prox_diam",
    "prox_major_axis",
    "prox_minor_axis",
    "prox_area",
    "mid_diam",
    "mid_major_axis",
    "mid_minor_axis",
    "mid_area",
    "dist_diam",
    "dist_major_axis",
    "dist_minor_axis",
    "dist_area",
    "periaortic_total_cm3",
    "periaortic_ring_cm3",
    "periaortic_fat_cm3",
    "periaortic_fat_mean_hu",
    "periaortic_fat_stddev_hu",
]
stats_output_keys = [
    "nifti_path",
    "ct_id",
    "mrn",
    "accession",
    "series_name",
    "nii_file_name",
    "total_completed",
    "contrast_completed",
    "aorta_completed",
    "tissues_completed",
    "ct_direction",
    "image_type",
    "sex",
    "height_at_scan_m",
    "weight_at_scan_kg",
    "pregnancy_status",
    "birthday",
    "scan_date",
    "age_at_scan",
    "length_mm",
    "width_mm",
    "slice_thickness_mm",
    "num_rows",
    "num_columns",
    "manufacturer",
    "manufacturer_model",
    "kvp",
    "sequence_name",
    "protocol_name",
    "contrast_pred",
    "contrast_prob",
    "contrast_bolus_agent",
    "contrast_bolus_route",
    "multienergy_ct",
    "procedure_code",
    "procedure_desc",
    "study_uid",
    "series_uid",
    "is_mip",
    "spleen_volume_cm3",
    "spleen_average_intensity",
    "kidney_right_volume_cm3",
    "kidney_right_average_intensity",
    "kidney_left_volume_cm3",
    "kidney_left_average_intensity",
    "gallbladder_volume_cm3",
    "gallbladder_average_intensity",
    "liver_volume_cm3",
    "liver_average_intensity",
    "stomach_volume_cm3",
    "stomach_average_intensity",
    "pancreas_volume_cm3",
    "pancreas_average_intensity",
    "adrenal_gland_right_volume_cm3",
    "adrenal_gland_right_average_intensity",
    "adrenal_gland_left_volume_cm3",
    "adrenal_gland_left_average_intensity",
    "lung_upper_lobe_left_volume_cm3",
    "lung_upper_lobe_left_average_intensity",
    "lung_lower_lobe_left_volume_cm3",
    "lung_lower_lobe_left_average_intensity",
    "lung_upper_lobe_right_volume_cm3",
    "lung_upper_lobe_right_average_intensity",
    "lung_middle_lobe_right_volume_cm3",
    "lung_middle_lobe_right_average_intensity",
    "lung_lower_lobe_right_volume_cm3",
    "lung_lower_lobe_right_average_intensity",
    "esophagus_volume_cm3",
    "esophagus_average_intensity",
    "trachea_volume_cm3",
    "trachea_average_intensity",
    "thyroid_gland_volume_cm3",
    "thyroid_gland_average_intensity",
    "small_bowel_volume_cm3",
    "small_bowel_average_intensity",
    "duodenum_volume_cm3",
    "duodenum_average_intensity",
    "colon_volume_cm3",
    "colon_average_intensity",
    "bladder_volume_cm3",
    "bladder_average_intensity",
    "prostate_volume_cm3",
    "prostate_average_intensity",
    "sacrum_volume_cm3",
    "sacrum_average_intensity",
    "vertebrae_S1_volume_cm3",
    "vertebrae_S1_average_intensity",
    "vertebrae_L5_volume_cm3",
    "vertebrae_L5_average_intensity",
    "vertebrae_L4_volume_cm3",
    "vertebrae_L4_average_intensity",
    "vertebrae_L3_volume_cm3",
    "vertebrae_L3_average_intensity",
    "vertebrae_L2_volume_cm3",
    "vertebrae_L2_average_intensity",
    "vertebrae_L1_volume_cm3",
    "vertebrae_L1_average_intensity",
    "vertebrae_T12_volume_cm3",
    "vertebrae_T12_average_intensity",
    "vertebrae_T11_volume_cm3",
    "vertebrae_T11_average_intensity",
    "vertebrae_T10_volume_cm3",
    "vertebrae_T10_average_intensity",
    "vertebrae_T9_volume_cm3",
    "vertebrae_T9_average_intensity",
    "vertebrae_T8_volume_cm3",
    "vertebrae_T8_average_intensity",
    "vertebrae_T7_volume_cm3",
    "vertebrae_T7_average_intensity",
    "vertebrae_T6_volume_cm3",
    "vertebrae_T6_average_intensity",
    "vertebrae_T5_volume_cm3",
    "vertebrae_T5_average_intensity",
    "vertebrae_T4_volume_cm3",
    "vertebrae_T4_average_intensity",
    "vertebrae_T3_volume_cm3",
    "vertebrae_T3_average_intensity",
    "vertebrae_T2_volume_cm3",
    "vertebrae_T2_average_intensity",
    "vertebrae_T1_volume_cm3",
    "vertebrae_T1_average_intensity",
    "vertebrae_C7_volume_cm3",
    "vertebrae_C7_average_intensity",
    "vertebrae_C6_volume_cm3",
    "vertebrae_C6_average_intensity",
    "vertebrae_C5_volume_cm3",
    "vertebrae_C5_average_intensity",
    "vertebrae_C4_volume_cm3",
    "vertebrae_C4_average_intensity",
    "vertebrae_C3_volume_cm3",
    "vertebrae_C3_average_intensity",
    "vertebrae_C2_volume_cm3",
    "vertebrae_C2_average_intensity",
    "vertebrae_C1_volume_cm3",
    "vertebrae_C1_average_intensity",
    "heart_volume_cm3",
    "heart_average_intensity",
    "aorta_volume_cm3",
    "aorta_average_intensity",
    "pulmonary_vein_volume_cm3",
    "pulmonary_vein_average_intensity",
    "brachiocephalic_trunk_volume_cm3",
    "brachiocephalic_trunk_average_intensity",
    "subclavian_artery_right_volume_cm3",
    "subclavian_artery_right_average_intensity",
    "subclavian_artery_left_volume_cm3",
    "subclavian_artery_left_average_intensity",
    "carotid_artery_right_volume_cm3",
    "carotid_artery_right_average_intensity",
    "carotid_artery_left_volume_cm3",
    "carotid_artery_left_average_intensity",
    "brachiocephalic_vein_left_volume_cm3",
    "brachiocephalic_vein_left_average_intensity",
    "brachiocephalic_vein_right_volume_cm3",
    "brachiocephalic_vein_right_average_intensity",
    "atrial_appendage_left_volume_cm3",
    "atrial_appendage_left_average_intensity",
    "superior_vena_cava_volume_cm3",
    "superior_vena_cava_average_intensity",
    "inferior_vena_cava_volume_cm3",
    "inferior_vena_cava_average_intensity",
    "portal_and_splenic_vein_volume_cm3",
    "portal_and_splenic_vein_average_intensity",
    "iliac_artery_left_volume_cm3",
    "iliac_artery_left_average_intensity",
    "iliac_artery_right_volume_cm3",
    "iliac_artery_right_average_intensity",
    "iliac_vena_left_volume_cm3",
    "iliac_vena_left_average_intensity",
    "iliac_vena_right_volume_cm3",
    "iliac_vena_right_average_intensity",
    "vertebrae_S1_midline",
    *[f"vertebrae_L{i}_midline" for i in range(5, 0, -1)],
    "vertebrae_T12L1_midline",  # This is a special case for between the T12 and L1 for determining abdominal aorta size
    *[f"vertebrae_T{i}_midline" for i in range(12, 0, -1)],
    *[f"vertebrae_C{i}_midline" for i in range(7, 0, -1)],
    "aorta_length_mm",
    "aorta_tortuosity_index",
    "aorta_soam",
    "aorta_periaortic_total_cm3",
    "aorta_periaortic_ring_cm3",
    "aorta_periaortic_fat_cm3",
    "aorta_periaortic_fat_mean_hu",
    "aorta_periaortic_fat_stddev_hu",
    *[f"asc_aorta_{diam}" for diam in aorta_region_columns],
    *[f"aortic_arch_{diam}" for diam in aorta_region_columns],
    *[f"desc_aorta_{diam}" for diam in aorta_region_columns],
    *[f"up_abd_aorta_{diam}" for diam in aorta_region_columns],
    *[f"lw_abd_aorta_{diam}" for diam in aorta_region_columns],
    "abdominal_scan",
    "total_body_volume_cm3",
    "total_subq_fat_volume_cm3",
    "total_visc_fat_volume_cm3",
    "total_skel_muscle_volume_cm3",
    "abdominal_body_volume_cm3",
    "abdominal_subq_fat_volume_cm3",
    "abdominal_visc_fat_volume_cm3",
    "abdominal_skel_muscle_volume_cm3",
    "L1_body_border_ratio",
    "L1_body_area_cm2",
    "L1_body_total_perimeter_cm",
    "L1_body_ellipse_perimeter_cm",
    "L1_body_circle_perimeter_cm",
    "L1_subq_fat_area_cm2",
    "L1_visc_fat_area_cm2",
    "L1_skel_muscle_area_cm2",
    "L3_body_border_ratio",
    "L3_body_area_cm2",
    "L3_body_total_perimeter_cm",
    "L3_body_ellipse_perimeter_cm",
    "L3_body_circle_perimeter_cm",
    "L3_subq_fat_area_cm2",
    "L3_visc_fat_area_cm2",
    "L3_skel_muscle_area_cm2",
    "L5_body_border_ratio",
    "L5_body_area_cm2",
    "L5_body_total_perimeter_cm",
    "L5_body_ellipse_perimeter_cm",
    "L5_body_circle_perimeter_cm",
    "L5_subq_fat_area_cm2",
    "L5_visc_fat_area_cm2",
    "L5_skel_muscle_area_cm2",
]


midline_keys = [
    *[f"vertebrae_L{i}_midline" for i in range(5, 0, -1)],
    *[f"vertebrae_T{i}_midline" for i in range(12, 0, -1)],
    *[f"vertebrae_C{i}_midline" for i in range(7, 0, -1)],
]
