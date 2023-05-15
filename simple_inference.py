# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU if you are on a multi GPU server
import json
import os

import click
import clip
import torch
from PIL import Image

from MLP import MLP

# This script will predict the aesthetic score for this image file:


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
    sd = torch.load(opts.model)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    pil_image = Image.open(opts.image)

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        image_features = clip_model.encode_image(image)

    im_emb_arr = image_features.type(torch.float)

    with torch.inference_mode():
        prediction = model(im_emb_arr)

    try:
        with open(os.path.splitext(opts.model)[0]+".json", "rt") as f:
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
