# portions adapted from https://github.com/OATML/RHO-Loss

def dataset_with_index(cls):
    class DatasetWithIndex(cls):
        def __getitem__(self, idx):
            return (idx, *(super().__getitem__(idx)))
    return DatasetWithIndex
