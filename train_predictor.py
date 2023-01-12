import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from MLP import MLP
import click

import numpy as np

@click.command()
@click.option('--epoch',                  help='Number of Epochs',                     type=int, required=True)
@click.option('--out',                    help='Output file of model',                 type=str, required=True)

@click.option('--learning-rate',          help='Learning Rate',                                         type=float, default=0.001)
@click.option('--val-percent',            help='Percent of embeddings to use for validation',           type=float, default=0.05)
@click.option('--batch-size',             help='Batch size',                                            type=int, default=256)
@click.option('--num-workers',             help='Number of workers',                                    type=int, default=16)
@click.option('--embedding-file',         help='Name of embeddings file',                               type=str, default='embeddings/x_embeddings.npy')
@click.option('--score-file',             help='Name of score file',                                    type=str, default='embeddings/y_ratings.npy')
@click.option('--device',                 help='Torch device type (default uses cuda if avaliable)',    type=str, default='default', show_default=True)
@click.option('--out',                    help='Output directory', metavar='DIR',                       type=str, default='embeddings')


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# load the training data 

def main(**kwargs): 
    opts = dotdict(kwargs)

    x = np.load (opts.embedding_file)

    y = np.load (opts.score_file)

    val_percentage = opts.val_percent

    train_border = int(x.shape()[0] * (1 - val_percentage) )

    train_tensor_x = torch.Tensor(x[:train_border]) # transform to torch tensor
    train_tensor_y = torch.Tensor(y[:train_border])

    train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create your datset
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,  num_workers=opts.num_workers) # create your dataloader


    val_tensor_x = torch.Tensor(x[train_border:]) # transform to torch tensor
    val_tensor_y = torch.Tensor(y[train_border:])

    '''
    print(train_tensor_x.size())
    print(val_tensor_x.size())
    print( val_tensor_x.dtype)
    print( val_tensor_x[0].dtype)
    '''

    val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create your datset
    val_loader = DataLoader(val_dataset, batch_size=512,  num_workers=16) # create your dataloader




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opts.device != 'default':
      device = opts.device

    model = MLP(768,x,y).to(device)   # CLIP embedding dim is 768 for CLIP ViT L 14
    model.lr = opts.learning_rate
    optimizer = torch.optim.Adam(model.parameters()) 

    # choose the loss you want to optimze for
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()

    epochs = opts.batch_size

    model.train()
    best_loss =999
    save_name = opts.out


    for epoch in range(epochs):
        losses = []
        losses2 = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())


            optimizer.step()

            if batch_num % 1000 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
                #print(y)

        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        losses = []
        losses2 = []
        
        for batch_num, input_data in enumerate(val_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            lossMAE = criterion2(output, y)
            #loss.backward()
            losses.append(loss.item())
            losses2.append(lossMAE.item())
            #optimizer.step()

            if batch_num % 1000 == 0:
                print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, batch_num, loss.item()))
                print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, batch_num, lossMAE.item()))
                
                #print(y)

        print('Validation - Epoch %d | MSE Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        print('Validation - Epoch %d | MAE Loss %6.2f' % (epoch, sum(losses2)/len(losses2)))
        if sum(losses)/len(losses) < best_loss:
            print("Best MAE Val loss so far. Saving model")
            best_loss = sum(losses)/len(losses)
            print( best_loss ) 

            torch.save(model.state_dict(), save_name )


    torch.save(model.state_dict(), save_name)

    print( best_loss ) 

    print("training done")
    # inferece test with dummy samples from the val set, sanity check
    print( "inferece test with dummy samples from the val set, sanity check")
    model.eval()
    output = model(x[:5].to(device))
    print(output.size())
    print(output)

if __name__ == "__main__":
   main()