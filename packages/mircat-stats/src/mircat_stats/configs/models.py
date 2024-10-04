torch_model_configs = {
    "total": {
        "name": "total",
        "spacing": [1.5, 1.5, 1.5],
        "patch_size": [128, 128, 128],
        "percentile_005": -1017,
        "percentile_995": 882,
        "mean": -260.3240051269531,
        "std": 477.13525390625,
        "output_map": {
            "spleen": 1,
            "kidney_right": 2,
            "kidney_left": 3,
            "gallbladder": 4,
            "liver": 5,
            "stomach": 6,
            "pancreas": 7,
            "adrenal_gland_right": 8,
            "adrenal_gland_left": 9,
            "lung_upper_lobe_left": 10,
            "lung_lower_lobe_left": 11,
            "lung_upper_lobe_right": 12,
            "lung_middle_lobe_right": 13,
            "lung_lower_lobe_right": 14,
            "esophagus": 15,
            "trachea": 16,
            "thyroid_gland": 17,
            "small_bowel": 18,
            "duodenum": 19,
            "colon": 20,
            "bladder": 21,
            "prostate": 22,
            "sacrum": 23,
            "vertebrae_S1": 24,
            "vertebrae_L5": 25,
            "vertebrae_L4": 26,
            "vertebrae_L3": 27,
            "vertebrae_L2": 28,
            "vertebrae_L1": 29,
            "vertebrae_T12": 30,
            "vertebrae_T11": 31,
            "vertebrae_T10": 32,
            "vertebrae_T9": 33,
            "vertebrae_T8": 34,
            "vertebrae_T7": 35,
            "vertebrae_T6": 36,
            "vertebrae_T5": 37,
            "vertebrae_T4": 38,
            "vertebrae_T3": 39,
            "vertebrae_T2": 40,
            "vertebrae_T1": 41,
            "vertebrae_C7": 42,
            "vertebrae_C6": 43,
            "vertebrae_C5": 44,
            "vertebrae_C4": 45,
            "vertebrae_C3": 46,
            "vertebrae_C2": 47,
            "vertebrae_C1": 48,
            "heart": 49,
            "aorta": 50,
            "pulmonary_vein": 51,
            "brachiocephalic_trunk": 52,
            "subclavian_artery_right": 53,
            "subclavian_artery_left": 54,
            "carotid_artery_right": 55,
            "carotid_artery_left": 56,
            "brachiocephalic_vein_left": 57,
            "brachiocephalic_vein_right": 58,
            "atrial_appendage_left": 59,
            "superior_vena_cava": 60,
            "inferior_vena_cava": 61,
            "portal_and_splenic_vein": 62,
            "iliac_artery_left": 63,
            "iliac_artery_right": 64,
            "iliac_vena_left": 65,
            "iliac_vena_right": 66,
        },
    },
    "body": {
        "name": "body",
        "spacing": [1.5, 1.5, 1.5],
        "patch_size": [128, 128, 128],
        "percentile_005": -985,
        "percentile_995": 1411,
        "mean": -43.4468879699707,
        "std": 355.778564453125,
        "output_map": {"body": 1},
    },
    "tissues": {
        "name": "tissues",
        "spacing": [1.5, 1.5, 1.5],
        "patch_size": [128, 128, 128],
        "percentile_005": -206,
        "percentile_995": 154,
        "mean": -33.02885055541992,
        "std": 74.95191192626953,
        "output_map": {"subq_fat": 1, "visc_fat": 2, "skel_muscle": 3},
    },
}
# Calculate the crop threshold using model vals
for config in torch_model_configs.values():
    config["crop_threshold"] = (config["percentile_005"] - config["mean"]) / config[
        "std"
    ]

nnunet_model_configs = {
    # Fill in configs here later
}
