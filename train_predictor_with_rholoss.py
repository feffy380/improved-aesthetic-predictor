# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import json
from functools import partial

import click
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import utils
from MLP import ILModel, RLossModel


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
@click.option(
    "--select-factor",
    help="Proportion of training data selected by RHO-Loss",
    type=float,
    default=0.1,
)
@click.option("--batch-size", help="Batch size", type=int, default=256)
@click.option("--val-batch-size", help="Validation batch size", type=int, default=1024)
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
@click.option(
    "--raw-scores", help="Use raw scores instead of normalizing them", is_flag=True
)
def main(**kwargs):
    opts = dotdict(kwargs)

    # ensure fixed seed for reproducible DataLoader order for RHO-Loss
    pl.seed_everything(opts.seed, workers=True)
    dataset_seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # if opts.device == "default" and torch.cuda.is_available():
    #     torch.set_float32_matmul_precision("high")
    if opts.device == "default" and torch.cuda.is_available():
        opts.device = "cuda"

    x = np.load(opts.embedding_file)
    y = np.load(opts.score_file)
    if len(x) != len(y):
        raise ValueError(
            f"Embedding and score file lengths don't match: {len(x)} vs {len(y)}"
        )
    # normalize ratings
    y_norm = y
    if not opts.raw_scores:
        y_norm = (y - y.mean()) / y.std()

    dataset = utils.dataset_with_index(TensorDataset)(
        torch.Tensor(x), torch.Tensor(y_norm)
    )

    # irreducible loss model
    print("Training irreducible loss model")
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [1 - opts.val_percent, opts.val_percent],
        torch.Generator().manual_seed(dataset_seed),
    )
    train_loader = DataLoader(
        val_dataset,  # IL model trains on the validation set
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=opts.val_batch_size,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )

    ilmodel = ILModel(x.shape[1], opts.learning_rate)  # input size = embedding length
    patience = 20
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="models/rhoLoss-il",
            filename=f"il-{opts.out}",
            save_weights_only=True,
            verbose=True,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=opts.epochs,
        callbacks=callbacks,
    )
    trainer.fit(ilmodel, train_loader, val_loader)
    ilmodel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, input_size=x.shape[1]
    )
    ilmodel.to(opts.device)
    ilmodel.eval()

    print("Precomputing irreducible losses")
    irreducible_loss = utils.compute_losses(dataloader=val_loader, model=ilmodel)

    print("Training reducible loss model")
    model = RLossModel(
        x.shape[1],
        lr=opts.learning_rate,
        selection_method=partial(
            utils.rho_loss_select,
            irreducible_loss=irreducible_loss,
            select_factor=opts.select_factor,
        ),
    )  # input size = embedding length

    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opts.val_batch_size,
        num_workers=opts.num_workers,
        persistent_workers=True,
    )

    trainer = pl.Trainer(
        max_epochs=opts.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
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

    # inference test sanity check
    print("inference test sanity check")
    model.eval()
    y_hat = model(torch.Tensor(x[:10]))
    y_target = y_norm[:10]
    print(y_hat)
    print(y_target)
    # rating mean and stddev in case we want to recover unstandardized ratings
    if not opts.raw_scores:
        with open(f"models/{opts.out}.json", "wt") as f:
            json.dump({"mean": str(y.mean()), "std": str(y.std())}, f)


if __name__ == "__main__":
    main()
