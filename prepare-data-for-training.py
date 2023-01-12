
# This script prepares the training images and ratings for the training.
# It assumes that all images are stored as files that PIL can read.
# It also assumes that the paths to the images files and the average ratings are in a .parquet files that can be read into a dataframe ( df ).

import pandas as pd
import statistics
#from torch.utils.data import Dataset, DataLoader
#import clip
#import torch
import click
from PIL import Image, ImageFile
import numpy as np
import time
import os

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)





@click.command()
@click.option('--score-path',             help='Training data', metavar='[DIR]',                    type=str, required=True)


@click.option('--filepath-col',        help='Column name for the images path in the dataframe', metavar='STR',type=str, default='IMAGEPATH')
@click.option('--scores-col',          help='Column name for the scores in the dataframe', metavar='STR',     type=str, default='AVERAGE_SCORE')
@click.option('--score-file-type',        help='Score file type',                                       type=click.Choice(['parquet', 'csv', 'json']), default='parquet', show_default=True)
@click.option('--data-dir',         help='Training data', metavar='[DIR]',                            type=str, default='')
@click.option('--embedding-name',         help='Name of embeddings file', metavar='STR',                type=str, default='x_embeddings')
@click.option('--score-name',             help='Name of score file', metavar='STR',                     type=str, default='y_ratings')
@click.option('--device',                 help='Torch device type (default uses cuda if avaliable)',    type=str, default='default', show_default=True)
@click.option('--clip',                   help='Model used by clip to embed images',                    type=str, default='ViT-L/14', show_default=True)
@click.option('--out',                    help='Output directory', metavar='DIR',                       type=str, default='embeddings')


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_df(df_type, df_name):
   if df_type == 'parquet':
      return pd.read_parquet(df_name)
   elif df_type == 'csv':
      return pd.read_csv(df_name)
   elif df_type == 'json':
      return pd.read_json(df_name)
   return None


def main(**kwargs):
   opts = dotdict(kwargs)
   outExists = os.path.exists(opts.out)
   if not outExists:
      os.makedirs(opts.out)   
   device = "cuda" if torch.cuda.is_available() else "cpu"
   if opts.device != 'default':
      device = opts.device
   
   model, preprocess = clip.load(opts.clip, device=device)

   
   f = opts.score_path

   df = load_df(opts.score_file_type, opts.score_path)


   x = []
   y = []
   c= 0

   for idx, row in df.iterrows():

      average_rating = float(row.opts.rating_column)
      print(average_rating)
      if average_rating <1:
         continue
      img= os.path.join(opts.data_dir, row.opts.filepath_column)     #assumes that the df has the column IMAGEPATH
      print(img)
      try:
         image = preprocess(Image.open(img)).unsqueeze(0).to(device)
      except:
   	   continue
      with torch.no_grad():
         image_features = model.encode_image(image)
      im_emb_arr = image_features.cpu().detach().numpy() 
      x.append(normalized ( im_emb_arr) )      # all CLIP embeddings are getting normalized. This also has to be done when inputting an embedding later for inference
      y_ = np.zeros((1, 1))
      y_[0][0] = average_rating
      #y_[0][1] = stdev      # I initially considered also predicting the standard deviation, but then didn't do it
      y.append(y_)
      print(c)
      c+=1
      x = np.vstack(x)
      y = np.vstack(y)
      print(x.shape)
      print(y.shape)
      np.save("{0}/{1}.npy".format(opts.out,opts.embeddings_name), x)
      np.save("{0}/{1}.npy".format(opts.out,opts.score_name), y)


if __name__ == "__main__":
   main()