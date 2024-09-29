import os
from typing import Callable
from copy import deepcopy
import torch
import pandas as pd
from torch.utils.data import Dataset

from exfiltration.my_tokenizers import CharacterTokenzier

class DNSExfiltrationDataset(Dataset):

    def __init__(self,
                 paths: str | list[str],
                 tokenizer: CharacterTokenzier,
                 preprocess_fns: list[list[Callable]] = [],
                 preprocess_params: list[list[dict]] = [],
                 *args, **kwargs) -> None:

        super().__init__()

        self.tokenizer = tokenizer
        
        print(f'Opening the following datasets: {", ".join(paths)}')
        datasets = [
            {
                'negative': pd.read_csv(os.path.join(path, 'negative.csv'), dtype={0: str, 1: bool}),
                'positive': pd.read_csv(os.path.join(path, 'positive.csv'), dtype={0: str, 1: bool}),
            } for path in paths]

        for i, (fns, params_list) in enumerate(zip(preprocess_fns, preprocess_params)):
            for fn, params in zip(fns, params_list):
                for k, v in datasets[i].items():
                    datasets[i][k] = fn(v, **params)

        data = pd.concat(
            (d for dataset in datasets for d in dataset.values()),
            axis='index'
        ).sample(frac=1)

        self.features = self.tokenizer(data['Subdomain'].astype(str).tolist())
        self.labels = torch.tensor(data['Exfiltration'].tolist(), dtype=torch.int64)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
class DNSExfiltrationMLM(Dataset):

    def __init__(self,
                 paths: list[str],
                 tokenizer: CharacterTokenzier,
                 mask_prob: float,
                 ignore_index: int,
                 preprocess_fns: list[list[Callable]] = [],
                 preprocess_params: list[list[dict]] = [],
                 *args,
                 **kwargs) -> None:
                 
        super().__init__()

        self.tokenizer = tokenizer

        print(f'Reading the following datasets: {", ".join(paths)}')
        datasets = [
            {
                'negative': pd.read_csv(os.path.join(path, 'negative.csv'), dtype={0: str, 1: bool}, na_filter=False),
                'positive': pd.read_csv(os.path.join(path, 'positive.csv'), dtype={0: str, 1: bool}, na_filter=False),
            } for path in paths]

        for i, (fns, params_list) in enumerate(zip(preprocess_fns, preprocess_params)):
            for fn, params in zip(fns, params_list):
                for k, v in datasets[i].items():
                    datasets[i][k] = fn(v, **params)

        data = pd.concat(
            (d for dataset in datasets for d in dataset.values()),
            axis='index'
        ).sample(frac=1)

        ids = self.tokenizer(data['Subdomain'].to_list(), pad = False, as_tensor = False)
        flatten_ids = [ii for i in ids for ii in i]
        self.features = torch.Tensor(flatten_ids).long()
        last_chunk_size = len(self.features) % tokenizer.max_seq_len
        self.features = self.features[:-last_chunk_size]
        self.features = self.features.reshape(-1, tokenizer.max_seq_len)

        # TODO: Do this dynamically
        random_mtx = torch.rand(self.features.shape)
        self.change_mask = random_mtx < mask_prob
        random_mtx = torch.rand(self.features.shape)
        mask_mask = random_mtx < 0.8
        random_mask = (random_mtx >= 0.8) & (random_mtx < 0.9)

        self.labels = deepcopy(self.features)
        self.labels[~self.change_mask] = ignore_index

        # Change 80% of selected tokens to mask token
        self.features[self.change_mask & mask_mask] = self.tokenizer.mask_token_idx

        # Change 10% of selected tokens to random tokens
        random_mtx = torch.randint(self.tokenizer.learned_tokens_start_idx, self.tokenizer.vocab_size, self.features.shape)
        self.features[self.change_mask & random_mask] = random_mtx[self.change_mask & random_mask]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]