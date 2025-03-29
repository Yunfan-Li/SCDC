import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from torch import Tensor


def get_dataset(data):
    print("=======Loading Data=======")
    notation = pd.read_table('./datasets/' + data + '_notations.txt',
                             sep='\t',
                             dtype=str)
    gene_names = pd.read_csv('./datasets/' + data + '_gene_names.txt',
                             sep=' ',
                             dtype=str,
                             header=None)

    batch_indices = notation['batch'].value_counts().keys().values
    batch_dict = {}
    n_batch = len(batch_indices)
    for (i, batch_index) in enumerate(batch_indices):
        notation.replace(batch_index, i, inplace=True)
        batch_dict[i] = batch_index
        print("Replace batch", batch_index, "with", i)
    batch_indices = torch.from_numpy(notation['batch'].values).long()

    cell_types = notation['celltype'].value_counts().keys().values
    cell_type_dict = {}
    n_type = len(cell_types)
    for (i, cell_type) in enumerate(cell_types):
        notation.replace(cell_type, i, inplace=True)
        cell_type_dict[i] = cell_type
        print("Replace cell type", cell_type, "with", i)
    cell_types = torch.from_numpy(notation['celltype'].values).long()
    idx, counts = torch.unique(cell_types, return_counts=True)
    print("=======Finish=======\n")

    print("=======Loading Data=======")
    sparse_counts = sparse.load_npz('./datasets/' + data +
                                    '_sparse_counts.npz')
    raw_counts = sparse_counts.todense()

    adata = sc.AnnData(raw_counts)
    raw_counts = torch.from_numpy(raw_counts).float()
 
    print("Preprocessing count data...")
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
    size_factors = torch.tensor(
        (adata.obs.n_counts / 10000).tolist()).view(-1, 1)

    sc.pp.log1p(adata)
    idx = sc.pp.highly_variable_genes(adata, inplace=False)
    gene_idx = idx['highly_variable'].tolist()
    sc.pp.highly_variable_genes(adata, subset=True)
    raw_counts = raw_counts[:, gene_idx]

    gene_names = gene_names.loc[:, gene_idx]

    sc.pp.scale(adata)

    data = torch.from_numpy(adata.X).float()
    print("=======Finish=======\n")

    print("=======Summary=======")
    print("After preprocessing", data.shape[0], "cells and", data.shape[1],
          "genes are selected.")
    print("The data includes", n_type, "cell types from", n_batch, "batches.")

    return data, size_factors, raw_counts, cell_types, batch_indices, data.shape[
        0], data.shape[
            1], n_type, n_batch, cell_type_dict, batch_dict, gene_names


class TensorDataSetWithIndex(TensorDataset):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor):
        super(TensorDataSetWithIndex, self).__init__(*tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index


def get_dataloader(data,
                   size_factors,
                   raw_counts,
                   cell_types,
                   batch_indices,
                   batch_size=256):
    dataset = TensorDataSetWithIndex(data, size_factors, raw_counts,
                                     cell_types, batch_indices)
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader