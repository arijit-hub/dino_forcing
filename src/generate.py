from torchvision.utils import save_image
from tqdm import tqdm

from src.training_module import SiTTrainingModule
import torch
import os

out_dir = "/path/to/output"
os.makedirs(out_dir, exist_ok=True)

model = SiTTrainingModule.load_from_checkpoint(
    "/path/to/ckpt",
    strict=False,
    map_location="cuda:0",
    weights_only=False,
)

model.eval()

cfgs = [4.0]
seeds = [3407]
conds = [912]
for cfg in cfgs:
    for cond in conds:
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            samples = model.reverse_process(
                conditions=[cond] * 1,
                num_sampling_steps=50,
                use_ema=True,
                generator=generator,
                cfg=cfg,
                return_self_condition=False,
            )
            save_image(
                samples,
                f"{out_dir}/sample_cond{cond}_{seed}_{cfg}.png",
                nrow=2,
            )
