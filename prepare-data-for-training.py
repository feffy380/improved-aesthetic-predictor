# This script prepares the training images and ratings for the training.
# It assumes that all images are stored as files that PIL can read.
# It also assumes that the paths to the images files and the average ratings are in a file pandas can import.

import os

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

Image.MAX_IMAGE_PIXELS = None


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_df(df_type, df_name):
    if df_type == "parquet":
        return pd.read_parquet(df_name)
    elif df_type == "csv":
        return pd.read_csv(df_name)
    elif df_type == "json":
        return pd.read_json(df_name)
    return None


def collate_discard_none(batch):
    return default_collate([sample for sample in batch if sample is not None])


class ImageDataset(Dataset):
    def __init__(
        self,
        img_ratings: pd.DataFrame,
        id_col="POSTID",
        img_col="IMAGEPATH",
        score_col="SCORE",
        transform=None,
        target_transform=None,
        flip: bool = False,
    ):
        self.img_ratings = img_ratings
        self.id_col = id_col
        self.img_col = img_col
        self.score_col = score_col
        self.transform = transform
        self.target_transform = target_transform
        self.flip = flip

    def __len__(self):
        n = len(self.img_ratings)
        if self.flip:
            n *= 2
        return n

    def __getitem__(self, idx):
        flipped = False
        if self.flip:
            flipped = idx % 2 == 1
            idx = idx // 2
        post_id = self.img_ratings[self.id_col].iloc[idx]
        img_path = self.img_ratings[self.img_col].iloc[idx]
        try:
            image = Image.open(img_path)
        except Exception:
            print(f"Couldn't load {img_path}")
            return None
        if flipped:
            image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        rating = float(self.img_ratings[self.score_col].iloc[idx])
        if self.transform:
            try:
                image = self.transform(images=image, return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)
            except Exception:
                print(f"Couldn't load {img_path}")
                return None
        if self.target_transform:
            rating = self.target_transform(rating)
        return post_id, image, rating


@click.command()
@click.option(
    "--score-file", help="Training data", metavar="[DIR]", type=str, required=True
)
@click.option(
    "--imagepath-col",
    help="Column name for the images path in the dataframe",
    metavar="STR",
    type=str,
    default="IMAGEPATH",
)
@click.option(
    "--score-col",
    help="Column name for the scores in the dataframe",
    metavar="STR",
    type=str,
    default="SCORE",
)
@click.option(
    "--score-file-type",
    help="Score file type",
    type=click.Choice(["parquet", "csv", "json"]),
    default="parquet",
    show_default=True,
)
@click.option(
    "--embedding-name",
    help="Name of embeddings file",
    metavar="STR",
    type=str,
)
@click.option(
    "--score-name",
    help="Name of score file",
    metavar="STR",
    type=str,
)
@click.option(
    "--device",
    help="Torch device type (default uses cuda if avaliable)",
    type=str,
    default="default",
    show_default=True,
)
@click.option(
    "--clip",
    help="Huggingface model used by clip to embed images",
    type=str,
    default="openai/clip-vit-large-patch14",
    show_default=True,
)
@click.option(
    "--out", help="Output directory", metavar="DIR", type=str, default="embeddings"
)
@click.option(
    "--flip-aug",
    help="Augment data by encoding horizontally flipped versions of images",
    is_flag=True,
)
def main(**kwargs):
    opts = dotdict(kwargs)
    outExists = os.path.exists(opts.out)
    if not outExists:
        os.makedirs(opts.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if opts.device != "default":
        device = opts.device
    basename = os.path.basename(opts.score_file)
    basename = os.path.splitext(basename)[0]
    if not opts.embeddings_name:
        opts.embeddings_name = f"x_{basename}_embeddings"
    if not opts.score_name:
        opts.score_name = f"y_{basename}_ratings"

    model = CLIPModel.from_pretrained(opts.clip).to(device)
    preprocess = AutoProcessor.from_pretrained(opts.clip)

    df = load_df(opts.score_file_type, opts.score_file)
    dataset = ImageDataset(img_ratings=df, transform=preprocess, flip=opts.flip_aug)

    post_ids = []
    x = []
    y = []

    with torch.inference_mode():
        for ids, images, ratings in tqdm(
            DataLoader(
                dataset, batch_size=64, collate_fn=collate_discard_none, num_workers=8
            )
        ):
            features = model.get_image_features(images.to(device))
            post_ids.append(ids)
            x.append(features)
            y.append(ratings)
    post_ids = torch.cat(post_ids).cpu().numpy()
    x = torch.cat(x).cpu().numpy()
    y = torch.cat(y).cpu().numpy()
    x = np.vstack(x)
    y = np.vstack(y)
    print(post_ids.shape)
    print(x.shape)
    print(y.shape)
    np.save(f"{opts.out}/ids_{basename}.npy", post_ids)
    np.save(f"{opts.out}/{opts.embeddings_name}.npy", x)
    np.save(f"{opts.out}/{opts.score_name}.npy", y)


if __name__ == "__main__":
    main()
