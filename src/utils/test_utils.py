import os

current_dir = os.path.dirname(os.path.abspath(__file__))


REF_IMAGES = {
    "imagenet_adm_256": f"{current_dir}/assets/VIRTUAL_imagenet256_labeled.npz",
    "jit_imagenet_256": f"{current_dir}/assets/jit_in256_stats_with_images.npz",
    "imagenet_adm_512": f"{current_dir}/assets/VIRTUAL_imagenet512.npz",
}

REF_CAPTION_FILES = {
    "imagenet_class": f"{current_dir}/assets/imagenet_50k_class_indices_for_generation.txt",
}

GEN_CAPTION_FILES = {
    "imagenet_class": f"{current_dir}/assets/imagenet_50k_class_indices_for_generation.txt",
}

REF_NUM_IMAGES = {
    "imagenet_adm_256": 50000,
    "jit_imagenet_256": 50000,
    "imagenet_adm_512": 50000,
}
