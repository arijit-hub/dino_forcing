"""Implements the data loading pipeline for DINO forcing."""

import torch
import numpy as np
import lightning.pytorch as L
import webdataset as wds
import os
import random
from glob import glob
from torch.distributed import get_world_size
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.utils.imagenet_classes import imagenet_class_to_idx
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


Image.MAX_IMAGE_PIXELS = None

############################################################
#                       LATENT SPACE                       #
############################################################


############################################################
#                    Transform functions                   #
############################################################
def cls_transform(
    data,
    img_size=256,
    num_classes=1_000,
    train=True,
    load_dino=True,
):
    """Transform the data necessarily."""

    mean = data[f"vae_embeddings_mean_{img_size}.npy"]
    std = data[f"vae_embeddings_std_{img_size}.npy"]
    json_data = data["json"]

    img_latent = mean if train == False else mean + np.random.randn(*std.shape) * std

    condition = json_data["class_idx"]
    if condition is None:
        condition = num_classes

    if load_dino:
        alignment_features = data["dino_features.npy"]
        return (img_latent, condition, alignment_features)
    return (img_latent, condition)


############################################################
#                      Collate functions                   #
############################################################
def latent_collate_fn(batch):

    ## Unpack the batch ##
    unpacked_batch = list(zip(*batch))

    ## We expect the images to be the first element ##
    images = unpacked_batch[0]

    ## Then its the class labels ##
    class_labels = unpacked_batch[1]

    ## Making the batch ##
    batch = {}
    batch["clean_imgs"] = torch.as_tensor(
        np.stack(images), dtype=torch.float
    ).contiguous()
    batch["conditions"] = torch.tensor(class_labels, dtype=torch.int).contiguous()

    try:
        alignment_features = unpacked_batch[2]
        batch["alignment_features"] = torch.as_tensor(
            np.stack(alignment_features), dtype=torch.float
        ).contiguous()
    except:
        pass

    return batch


############################################################
#       Lightning latent datamodule for class cond         #
############################################################
class LatentDataModule(L.LightningDataModule):
    """Base DataModule for any tar dataset with latent variables (unconditional by default)."""

    def __init__(
        self,
        train_root_dir: str,
        val_root_dir: str,
        train_dataset_size=None,  # 1281167,
        val_dataset_size=None,  # 50000,
        batch_size: int = 32,
        num_workers: int = 8,
        img_size: int = 256,
        buffer_size: int = 1000,
        train_transform_fn=None,
        val_transform_fn=None,
        collate_fn=None,
    ):
        """Constructor."""
        super().__init__()

        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir

        ## Setting the dataset size ##
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size

        assert (
            self.train_dataset_size is not None
        ), "train_dataset_size must be provided."

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.buffer_size = buffer_size

        assert train_transform_fn is not None, "train_transform_fn must be provided."
        assert val_transform_fn is not None, "val_transform_fn must be provided."
        self.train_transform_fn = train_transform_fn
        self.val_transform_fn = val_transform_fn

        assert collate_fn is not None, "collate_fn must be provided."
        self.collate_fn = collate_fn

    def _make_dataset(self, train: bool = True):
        """Creates a webdataset dataset."""

        ## Fetching the urls ##
        if train:
            dir = self.train_root_dir.split(" ")
        else:
            dir = self.val_root_dir.split(" ")

        urls = []
        for d in dir:
            urls.extend(sorted(glob(os.path.join(d, "*.tar"))))
        random.seed(3407)  ## Setting the seed for reproducibility
        random.shuffle(urls)  ## Shuffle the urls to ensure randomness across workers

        ## Creating the dataset ##
        dataset = (
            wds.WebDataset(
                urls,
                shardshuffle=1 if train else 0,  ## Shuffle the shards for workers
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            .shuffle(
                self.buffer_size if train else False,
            )  ## Create a buffer of samples from the workers and shuffle them
            .decode()  ## Decode the data
            .map(
                self.train_transform_fn if train else self.val_transform_fn
            )  ## Transforming the data
            .batched(
                self.batch_size,
                collation_fn=self.collate_fn,
                partial=False,
            )  ## Already batch samples from the tar files
        )

        return dataset

    def _make_loader(self, train: bool = True):
        """Creates a webdataset loader."""

        ## Creating the dataloader ##
        loader = (
            wds.WebLoader(
                self.train_dataset if train else self.val_dataset,
                num_workers=self.num_workers if train else 2,
                batch_size=None,
                persistent_workers=True,
            )
            .with_length(
                (self.train_dataset_size if train else self.val_dataset_size)
                // (self.batch_size * get_world_size())
            )
            .with_epoch(
                (self.train_dataset_size if train else self.val_dataset_size)
                // (self.batch_size * get_world_size())
            )
        )

        return loader

    def setup(self, stage=None):
        """Setup the data."""

        ## Creating the datasets ##
        self.train_dataset = self._make_dataset(train=True)
        self.val_dataset = self._make_dataset(train=False)

    def train_dataloader(self):
        """Returns the train data loader."""
        return self._make_loader(train=True)

    def val_dataloader(self):
        """Returns the val data loader."""
        return self._make_loader(train=False)

############################################################
#                       PIXEL SPACE                        #
############################################################

############################################################
#                    Transform functions                   #
############################################################
def imagenet_transform(data, load_dino=False, img_size=256):
    """Transform the data necessarily."""

    aug = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data["clean_imgs"] = aug(data["img"])

    img_path = data.get("img_path", None)
    if img_path is not None:
        data["conditions"] = imagenet_class_to_idx.get(Path(img_path).parent.name, 1000)

    else:
        data["conditions"] = 1000

    if load_dino:
        dino_aug = transforms.Compose(
            [
                transforms.Resize(
                    224 * (img_size // 256),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(224 * (img_size // 256)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        data["alignment_features"] = dino_aug(data["img"])

    del data["img"]
    del data["img_path"]

    return data

#############################################################
#                      Pixel Dataset                        #
#############################################################

class PixelDataset(torch.utils.data.Dataset):
    """Dataset for pixel-based datasets."""

    def __init__(
        self,
        root_dir: str,
        img_extension: str = "JPEG",
        transform_fn=None,
    ):
        """Constructor."""
        super().__init__()

        self.root_dir = root_dir
        self.img_extension = img_extension

        assert transform_fn is not None, "transform_fn must be provided."
        self.transform_fn = transform_fn

        ## Fetching the image paths ##
        self.image_paths = sorted(
            glob(f"{self.root_dir}/**/*.{self.img_extension}", recursive=True)
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns a sample from the dataset."""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        sample = {"img": img, "img_path": img_path}
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample

############################################################
#        Lightning pixel datamodule for class cond         #
############################################################
class PixelDataModule(L.LightningDataModule):
    """Datamodule for pixel-based datasets."""

    def __init__(
        self,
        train_root_dir: str,
        val_root_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        img_size: int = 256,
        img_extension: str = "JPEG",
        train_transform_fn=None,
        val_transform_fn=None,
        collate_fn=None,
    ):
        """Constructor."""
        super().__init__()

        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.img_extension = img_extension

        assert train_transform_fn is not None, "train_transform_fn must be provided."
        assert val_transform_fn is not None, "val_transform_fn must be provided."
        self.train_transform_fn = train_transform_fn
        self.val_transform_fn = val_transform_fn

        assert collate_fn is not None, "collate_fn must be provided."
        self.collate_fn = collate_fn

    def _make_dataset(self, train: bool = True):
        """Creates a pixel dataset."""
        if train:
            dataset = PixelDataset(
                root_dir=self.train_root_dir,
                img_extension=self.img_extension,
                transform_fn=self.train_transform_fn,
            )
        else:
            dataset = PixelDataset(
                root_dir=self.val_root_dir,
                img_extension=self.img_extension,
                transform_fn=self.val_transform_fn,
            )
        return dataset

    def _make_loader(self, train: bool = True):
        """Creates a pixel dataloader."""
        dataset = self.train_dataset if train else self.val_dataset
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers if train else 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
        return loader

    def setup(self, stage=None):
        """Setup the data."""
        self.train_dataset = self._make_dataset(train=True)
        self.val_dataset = self._make_dataset(train=False)
        self.train_dataset_size = len(self.train_dataset)
        self.val_dataset_size = len(self.val_dataset)

    def train_dataloader(self):
        """Returns the train data loader."""
        return self._make_loader(train=True)

    def val_dataloader(self):
        """Returns the val data loader."""
        return self._make_loader(train=False)
