# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU if you are on a multi GPU server
import json
import os

import click
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, logging

from MLP import MLP

logging.set_verbosity_error()

# This script will predict the aesthetic score for the given image file


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@click.command()
@click.option(
    "--image", help="Image file to evaluate", metavar="[DIR]", type=str, required=True
)
@click.option(
    "--model", help="Directory of model", metavar="[DIR]", type=str, required=True
)
@click.option(
    "--clip",
    help="Huggingface model used by clip to embed images",
    type=str,
    default="openai/clip-vit-large-patch14",
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

    clip_model = CLIPModel.from_pretrained(opts.clip).to(device)
    preprocess = AutoProcessor.from_pretrained(opts.clip)
    dim = clip_model.projection_dim

    model = MLP(dim)  # CLIP embedding dim is 768 for CLIP ViT L 14
    sd = torch.load(opts.model)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    pil_image = Image.open(opts.image)

    image = preprocess(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

    with torch.inference_mode():
        image_features = clip_model.get_image_features(image)

    im_emb_arr = image_features.type(torch.float)

    with torch.inference_mode():
        prediction = model(im_emb_arr)

    try:
        with open(os.path.splitext(opts.model)[0] + ".json", "rt") as f:
            y_stats = json.load(f)
    except Exception:
        y_stats = None
    print("Aesthetic score predicted by the model:")
    if y_stats is None:
        print(prediction.item())
    else:
        print(prediction.item() * float(y_stats["std"]) + float(y_stats["mean"]))


if __name__ == "__main__":
    main()
