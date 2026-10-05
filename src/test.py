"""Facilitates the training of models."""

import os
from pathlib import Path
import hydra
from hydra.utils import instantiate
import torch
from lightning.pytorch import seed_everything
from src.utils.test_utils import (
    REF_IMAGES,
    REF_CAPTION_FILES,
    GEN_CAPTION_FILES,
    REF_NUM_IMAGES,
)

from genmetrics import GenMetric, GenDataset
from torch.utils.data import DataLoader, Subset
from src.training_module import SiTTrainingModule
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("high")
seed_everything(3407, workers=True)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def test(cfg):
    """Main function which orchestrates everything."""

    ## Setting the genmetric module ##
    metric = GenMetric(
        metrics=cfg.metrics,
        feature_extractors_for_inception_metrics=cfg.feature_extractors_for_inception_metrics,
    ).to(torch.device("cuda"))

    ## Setting the dataset ##
    real_dataset = GenDataset(
        image_path=REF_IMAGES.get(cfg.dataset, None),
        return_no_images=cfg.return_no_images_for_real_dataloading,
        ref_caption_path=REF_CAPTION_FILES.get(cfg.ref_caption, None),
        gen_caption_path=GEN_CAPTION_FILES.get(cfg.gen_caption, None),
    )

    fake_dataset = GenDataset(
        image_path=None,
        ref_caption_path=REF_CAPTION_FILES.get(cfg.ref_caption, None),
        gen_caption_path=GEN_CAPTION_FILES.get(cfg.gen_caption, None),
    )

    if "imagenet_adm" not in cfg.dataset and "jit_imagenet" not in cfg.dataset:
        real_dataset = Subset(real_dataset, list(range(REF_NUM_IMAGES[cfg.dataset])))

    fake_dataset = Subset(fake_dataset, list(range(REF_NUM_IMAGES[cfg.dataset])))

    real_dl = DataLoader(
        real_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
    )

    fake_dl = DataLoader(
        fake_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
    )

    ## Loading the model ##
    model = SiTTrainingModule.load_from_checkpoint(
        cfg.ckpt_path,
        map_location="cuda",
        weights_only=False,
    ).to(torch.device("cuda"))

    model.eval()

    guiding_ckpt_path = cfg.get("guiding_ckpt_path", "")

    if model.network.use_self_cond:
        if len(guiding_ckpt_path) > 0:
            guiding_model = SiTTrainingModule.load_from_checkpoint(
                guiding_ckpt_path,
                map_location="cuda",
                weights_only=False,
            ).to(torch.device("cuda"))

            guiding_model.eval()

    else:
        guiding_model = None

    ## Setting the generator ##
    generator = torch.Generator(device="cuda").manual_seed(3407)

    logger = instantiate(cfg.trainer.logger)
    use_ema = cfg.use_ema if "use_ema" in cfg else True

    ## Running the evaluation ##
    for batch in fake_dl:
        conditions = batch["gen_caption"]
        conditions = [int(c) for c in conditions]

        images = model.reverse_process(
            conditions=conditions,
            num_sampling_steps=cfg.num_sampling_steps,
            cfg=cfg.cfg_scale,
            generator=generator,
            use_ema=use_ema,
            guiding_model=(
                guiding_model.ema_network if len(guiding_ckpt_path) > 0 else None
            ),
        )

        images = images * 255.0
        images = images.to(torch.uint8)

        metric.update(
            images=images,
            mean=batch["mean"].to(torch.device("cuda")),
            sigma=batch["sigma"].to(torch.device("cuda")),
            real=False,
            captions=conditions,
        )

    for batch in real_dl:
        metric.update(
            images=(
                batch["image"].to(torch.device("cuda")) if "image" in batch else None
            ),
            mean=batch["mean"].to(torch.device("cuda")) if "mean" in batch else None,
            sigma=batch["sigma"].to(torch.device("cuda")) if "sigma" in batch else None,
            real=True,
            captions=batch["ref_caption"] if "ref_caption" in batch else None,
        )

    results = metric.compute()

    postfix = f"_{cfg.postfix}" if "postfix" in cfg else ""
    online_postfix = f"_online" if not use_ema else ""
    postfix += online_postfix

    results = {
        f"test/{cfg.dataset}_{cfg.cfg_scale}{postfix}/{k}": v
        for k, v in results.items()
    }

    ## Logging the results ##
    logger.log_metrics(results, step=cfg.global_step)


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    test()
