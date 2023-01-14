
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
import pandas as pd
from MLP import MLP
import clip

import os

@click.option('--directory',                    help='Image directory to evaluate',               type=str, required=True)
@click.option('--model-path',                   help='Directory of model',                   type=str, required=True)


@click.option('--out',                          help='CSV output with scores',                   type=str, default='scores')
@click.option('--clip',                         help='Model used by clip to embed images',                    type=str, default='ViT-L/14', show_default=True)
@click.option('--device',                       help='Torch device type (default uses cuda if avaliable)',    type=str, default='default', show_default=True)
@click.option('--checkpoint',                   help='How often to save CSV',    type=int, default=100, show_default=True)


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
    scores = []
    count = 0
    for file in os.listdir(opts.directory):
        with torch.no_grad():
            pil_image = Image.open(file)
            image = preprocess(pil_image).unsqueeze(0).to(device)
            image_features = model2.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy() )
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            scores.append({'file':file, 'score':prediction.item()})
            print("{0}: {1}".format( prediction.item() ))
            if count % opts.checkpoint == 0:
                df = pd.DataFrame(scores)
                df.to_csv("{0}.csv".format(opts.out), index = False)

    df = pd.DataFrame(scores)
    df.to_csv("{0}.csv".format(opts.out), index = False)

if __name__ == "__main__":
   main()