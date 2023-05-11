# This script prepares the training images and ratings for the training.
# It assumes that all images are stored as files that PIL can read.
# It also assumes that the paths to the images files and the average ratings are in a file pandas can import.

import os

import click
import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

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
    ):
        self.img_ratings = img_ratings
        self.id_col = id_col
        self.img_col = img_col
        self.score_col = score_col
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ratings)

    def __getitem__(self, idx):
        post_id = self.img_ratings[self.id_col].iloc[idx]
        img_path = self.img_ratings[self.img_col].iloc[idx]
        try:
            image = Image.open(img_path)
        except Exception:
            print(f"Couldn't load {img_path}")
            return None
        rating = float(self.img_ratings[self.score_col].iloc[idx])
        if self.transform:
            try:
                image = self.transform(image)
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
    help="Model used by clip to embed images",
    type=str,
    default="ViT-L/14",
    show_default=True,
)
@click.option(
    "--out", help="Output directory", metavar="DIR", type=str, default="embeddings"
)
def main(**kwargs):
    opts = dotdict(kwargs)
    outExists = os.path.exists(opts.out)
    if not outExists:
        os.makedirs(opts.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if opts.device != "default":
        device = opts.device
    if not opts.embeddings_name:
        opts.embeddings_name = f"x_{os.path.splitext(opts.score_file)[0]}_embeddings"
    if not opts.score_name:
        opts.score_name = f"y_{os.path.splitext(opts.score_file)[0]}_ratings"

    model, preprocess = clip.load(opts.clip, device=device)

    df = load_df(opts.score_file_type, opts.score_file)
    dataset = ImageDataset(img_ratings=df, transform=preprocess)

    post_ids = []
    x = []
    y = []

    with torch.no_grad():
        for ids, images, ratings in tqdm(
            DataLoader(
                dataset, batch_size=64, collate_fn=collate_discard_none, num_workers=8
            )
        ):
            features = model.encode_image(images.to(device))
            post_ids.append(ids)
            x.append(features)
            y.append(ratings)
    post_ids = torch.cat(post_ids).cpu().numpy()
    x = torch.cat(x)
    x_fnorm = F.normalize(x, dim=-1).cpu().numpy()
    x = x.cpu().numpy()
    # x_norm = [normalized(xi) for xi in x]
    y = torch.cat(y).cpu().numpy()
    x = np.vstack(x)
    # x_norm = np.vstack(x_norm)
    x_fnorm = np.vstack(x_fnorm)
    y = np.vstack(y)
    print(post_ids.shape)
    print(x.shape)
    # print(x_norm.shape)
    print(x_fnorm.shape)
    print(y.shape)
    np.save(f"{opts.out}/ids_{os.path.splitext(opts.score_file)[0]}.npy", post_ids)
    np.save(f"{opts.out}/{opts.embeddings_name}.npy", x)
    # np.save(f"{opts.out}/{opts.embeddings_name}_norm.npy", x_norm)
    np.save(f"{opts.out}/{opts.embeddings_name}_fnorm.npy", x_fnorm)
    np.save(f"{opts.out}/{opts.score_name}.npy", y)


if __name__ == "__main__":
    main()
