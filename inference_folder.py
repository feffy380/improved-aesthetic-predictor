import os

import click
import clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from MLP import MLP

Image.MAX_IMAGE_PIXELS = None


def collate_discard_none(batch):
    return default_collate([sample for sample in batch if sample is not None])


class FolderDataset(Dataset):
    def __init__(
        self,
        img_dir,
        transform=None,
    ):
        self.img_dir = img_dir
        self.img_list = [
            img
            for img in os.listdir(self.img_dir)
            if os.path.splitext(img)[1] in (".jpg", ".png", ".gif")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        try:
            image = Image.open(img_path)
        except Exception:
            print(f"Couldn't load {img_path}")
            return None
        if self.transform:
            try:
                image = self.transform(image)
            except Exception:
                print(f"Couldn't load {img_path}")
                return None
        return img_path, image


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@click.command()
@click.option(
    "--directory", help="Image directory to evaluate", type=str, required=True
)
@click.option("--model", help="Directory of model", type=str, required=True)
@click.option("--out", help="CSV output with scores", type=str, default="scores")
@click.option(
    "--clip",
    help="Model used by clip to embed images",
    type=str,
    default="ViT-L/14",
    show_default=True,
)
@click.option(
    "--device",
    help="Torch device type (default uses cuda if avaliable)",
    type=str,
    default="default",
    show_default=True,
)
def main(**kwargs):
    opts = dotdict(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if opts.device != "default":
        device = opts.device

    clip_model, preprocess = clip.load(opts.clip, device=device)  # RN50x64
    dim = clip_model.visual.output_dim

    model = MLP(dim)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(
        opts.model
    )  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s.get("state_dict", s))
    model.to(device)
    model.eval()

    files = []
    embeds = []
    dataset = FolderDataset(img_dir=opts.directory, transform=preprocess)
    with torch.no_grad():
        for paths, images in tqdm(
            DataLoader(
                dataset, batch_size=64, collate_fn=collate_discard_none, num_workers=8
            )
        ):
            features = clip_model.encode_image(images.to(device))
            embeds.append(features)
            files.extend(paths)
    embeds = torch.cat(embeds)

    scores = []
    for path, embed in zip(files, embeds):
        with torch.no_grad():
            y_hat = model(embed.type(torch.float))
            scores.append({"file": path, "score": y_hat.item()})
            print(f"{path}: {y_hat.item()}")

    df = pd.DataFrame(scores)
    df.to_csv(f"{opts.out}.csv", index=False)


if __name__ == "__main__":
    main()
