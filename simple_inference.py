from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from MLP import MLP
import clip


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

@click.option('--image',                  help='Image file to evaluate', metavar='[DIR]',                    type=str, required=True)
@click.option('--model-path',             help='Directory of model', metavar='[DIR]',                    type=str, required=True)
@click.option('--clip',                   help='Model used by clip to embed images',                    type=str, default='ViT-L/14', show_default=True)
@click.option('--device',                 help='Torch device type (default uses cuda if avaliable)',    type=str, default='default', show_default=True)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def main(**kwargs):
    opts = dotdict(kwargs)
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if opts.device != 'default':
        device = opts.device

    s = torch.load(opts.model_path)   # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)

    model.to(device)
    model.eval()


    model2, preprocess = clip.load(opts.clip, device=device)  #RN50x64   


    pil_image = Image.open(opts.image)

    image = preprocess(pil_image).unsqueeze(0).to(device)



    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    print( "Aesthetic score predicted by the model:")
    print( prediction )


if __name__ == "__main__":
   main()