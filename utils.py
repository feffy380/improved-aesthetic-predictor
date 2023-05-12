# portions adapted from https://github.com/OATML/RHO-Loss
import torch
import torch.nn.functional as F


def dataset_with_index(cls):
    class DatasetWithIndex(cls):
        def __getitem__(self, idx):
            return (idx, *(super().__getitem__(idx)))

    return DatasetWithIndex


def compute_losses(dataloader, model, device=None):
    """Compute losses for full dataset.

    (I did not implement this with
    trainer.predict() because the trainset returns (index, x, y) and does not
    readily fit into the forward method of the irreducible loss model.)

    Returns:
        losses: Tensor, losses for the full dataset, sorted by <globa_index> (as
        returned from the train data loader). losses[global_index] is the
        irreducible loss of the datapoint with that global index. losses[idx] is
        nan if idx is not part of the dataset.
        targets: Tensor, targets of the datsets, sorted by <index> (as
        returned from the train data loader). Also see above.This is just used to verify that
        the data points are indexed in the same order as they were in this
        function call, when <losses> is used.
    """
    if device is None:
        device = model.device
    losses = []
    idxs = []
    with torch.inference_mode():
        for idx, x, target in dataloader:
            idx, x, target = idx.to(device), x.to(device), target.to(device)
            y_hat = model(x)
            loss = F.mse_loss(y_hat, target, reduction="none")
            losses.append(loss)
            idxs.append(idx)
    losses_temp = torch.cat(losses, dim=0)
    idxs = torch.cat(idxs, dim=0)
    max_idx = idxs.max()

    # fill missing values with nan
    losses = torch.tensor(
        [[float("nan")]] * (max_idx + 1), dtype=losses_temp.dtype, device=device
    )
    losses[idxs] = losses_temp

    return losses


def rho_loss_select(
    batch: list,
    model: torch.nn.Module,
    irreducible_loss: torch.Tensor,
    loss_function,
    select_factor: float,
):
    """
    Selects samples using reducible loss
    Args:
        batch: List[Tuple], list of samples in batch
        model: nn.Module, model being trained
        irreducible_loss: Tensor, irreducible losses for train set, ordered by global index
        loss_function: Callable, loss function to be used for RHO-Loss
        select_factor: float, proportion of batch to be selected
    Returns:
        selected_minibatch: list of selected samples with highest reducible loss
    """
    # unpack batch into separate lists
    idx, data, target = batch
    batch_size = len(data)
    selected_batch_size = max(1, int(batch_size * select_factor))
    assert selected_batch_size <= batch_size

    model.eval()
    with torch.no_grad():
        # RHO-Loss = batch_losses - irreducible_loss
        model_loss = loss_function(model(data), target, reduction="none")
        irreducible_loss.to(model_loss.device)
        reducible_loss = (model_loss - irreducible_loss[idx]).squeeze(dim=-1)
        selected_idxs = torch.argsort(reducible_loss, descending=True)[
            :selected_batch_size
        ]
    model.train()

    selected_minibatch = (
        idx[selected_idxs],
        data[selected_idxs],
        target[selected_idxs],
    )
    return selected_minibatch
