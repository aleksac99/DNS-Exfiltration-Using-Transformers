from typing import Literal
from torch.nn import Module
from torch.utils.data import Dataset

from .models import BertClassifier, BertMLM
from .datasets import DNSExfiltrationDataset, DNSExfiltrationMLM
from .my_tokenizers import CharacterTokenzier

def get_model_data_classes_from_task(task: Literal['mlm', 'finetuning']) -> tuple[Module, Dataset]:

    if task == 'mlm':
        return BertMLM, DNSExfiltrationMLM
    elif task == 'finetuning':
        return BertClassifier, DNSExfiltrationDataset
    else:
        raise ValueError("`task` must be either `mlm` or `finetuning`.")

def load_pretrained_tokenizer(tokenizer_config: dict) -> CharacterTokenzier:
    
    tokenizer = CharacterTokenzier.from_pretrained(tokenizer_config['path'])
    tokenizer.to_config(tokenizer_config)

    return tokenizer