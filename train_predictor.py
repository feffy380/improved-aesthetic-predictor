# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import json

import click
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from MLP import MLP


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# load the training data


@click.command()
@click.option("--epochs", help="Number of epochs", type=int, default=-1)
@click.option("--out", help="Output file of model", type=str, required=True)
@click.option("--learning-rate", help="Learning Rate", type=float, default=1e-3)
@click.option(
    "--val-percent",
    help="Percent of embeddings to use for validation",
    type=float,
    default=0.05,
)
@click.option("--batch-size", help="Batch size", type=int, default=256)
@click.option("--num-workers", help="Number of workers", type=int, default=16)
@click.option(
    "--embedding-file",
    help="Name of embeddings file",
    type=str,
    default="embeddings/x_embeddings.npy",
)
@click.option(
    "--score-file",
    help="Name of score file",
    type=str,
    default="embeddings/y_ratings.npy",
)
@click.option(
    "--device",
    help="Torch device type (default uses cuda if avaliable)",
    type=str,
    default="default",
    show_default=True,
)
@click.option(
    "--seed",
    help="random seed",
    type=int,
)
def main(**kwargs):
    opts = dotdict(kwargs)

    pl.seed_everything(opts.seed, workers=True)

    # if opts.device == "default" and torch.cuda.is_available():
    #     torch.set_float32_matmul_precision("high")

    x = np.load(opts.embedding_file)
    y = np.load(opts.score_file)
    if len(x) != len(y):
        raise ValueError(
            f"Embedding and score file lengths don't match: {len(x)} vs {len(y)}"
        )
    # normalize ratings
    y_norm = (y - y.mean()) / y.std()
    mean = y_norm.mean()
    print(np.mean(np.square(y_norm - mean)))

    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y_norm))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [1 - opts.val_percent, opts.val_percent]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )

    model = MLP(x.shape[1], opts.learning_rate)  # input size = embedding length
    trainer = pl.Trainer(
        max_epochs=opts.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="models",
                filename=opts.out,
                save_weights_only=True,
                verbose=True,
            ),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print("training done")

    # inference test with dummy samples from the val set, sanity check
    print("inference test with dummy samples from the val set, sanity check")
    model.eval()
    y_hat = model(torch.Tensor(x[:10]))
    y_target = y_norm[:10]
    print(y_hat)
    print(y_target)
    # rating mean and stddev in case we want to recover unstandardized ratings
    with open(f"models/{opts.out}.json", "wt") as f:
        json.dump({"mean": str(y.mean()), "std": str(y.std())}, f)


if __name__ == "__main__":
    main()
