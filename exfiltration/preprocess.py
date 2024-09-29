from typing import Callable
import numpy as np
import pandas as pd

def drop_duplicates(dataset: pd.DataFrame):
    dataset = dataset.drop_duplicates()
    return dataset

def select_subset(dataset: pd.DataFrame, p: float, shuffle: bool = False):
    
    idx = np.random.permutation(len(dataset)) if shuffle else np.arange(len(dataset))

    return dataset.iloc[idx[:int(len(dataset) * p)]] if isinstance(p, float) else dataset.iloc[idx[:p]]

def get_preprocess_fn(name):

    match name:
        case 'drop_duplicates':
            return drop_duplicates
        case 'select_subset':
            return select_subset
        case _:
            raise ValueError
        
def get_preprocess_fns(names: list[str] | list[list[str]]) -> list[Callable]:
        
        return [[get_preprocess_fn(name) for name in fn_names] for fn_names in names]