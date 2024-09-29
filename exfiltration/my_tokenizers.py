"""
Custom Tokenizer.
"""
import json

import torch
import torch.nn as nn


class CharacterTokenzier(nn.Module):

    def __init__(self,
                 chars: str,
                 max_seq_len: int,
                 cls_token: str = '[CLS]',
                 unk_token: str = '[UNK]',
                 pad_token: str = '[PAD]',
                 mask_token: str = '[MASK]',
                 sep_token: str = '[SEP]',
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.learned_tokens_start_idx = 5
        self.max_seq_len = max_seq_len

        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.sep_token = sep_token
        self._chars = chars

        self._tokens_to_ids = {
            cls_token: 0,
            mask_token: 1,
            pad_token: 2,
            unk_token: 3,
            sep_token: 4,
            **{token: self.learned_tokens_start_idx + idx for idx, token in enumerate(chars)}}
        
        self._ids_to_tokens = {idx: token for token, idx in self._tokens_to_ids.items()}

    @classmethod
    def from_pretrained(cls, path: str):
        
        with open(path, 'r') as f:
            tokenizer = json.load(f)

        return cls(**tokenizer)
    
    def to_file(self, path: str):
        with open(path, 'w') as f:
            
            json.dump({
                "max_seq_len": self.max_seq_len,
                "cls_token": self.cls_token,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
                "chars": self._chars
            }, f)

    def to_config(self, tokenizer_config: dict) -> None:

        tokenizer_config['type'] = 'character'
        tokenizer_config['pretrained'] = True
        tokenizer_config['chars'] = self._chars
        tokenizer_config['max_seq_len'] = self.max_seq_len
        tokenizer_config['special_tokens'] = {
            'cls_token': self.cls_token,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'mask_token': self.mask_token,
            'sep_token': self.sep_token
            }

    def __ids_to_tokens_sample(self, sample, remove_default_tokens):

        if remove_default_tokens:
            return ''.join([self._ids_to_tokens.get(idx.item(), self.unk_token) for idx in sample if idx.item() >= 3])
        else: 
            return ''.join([self._ids_to_tokens.get(idx.item(), self.unk_token) for idx in sample])
        
    def ids_to_tokens(self, ids: torch.Tensor, remove_default_tokens: bool):

        if ids.ndim==1:
            result = self.__ids_to_tokens_sample(ids, remove_default_tokens)
        elif ids.ndim==2:
            result = [
                self.__ids_to_tokens_sample(sample, remove_default_tokens)
                for sample in ids
            ]
        else:
            raise ValueError('`ids` should be either 1D or 2D tensor.')
        
        return result

    def forward(self, text: list, pad: bool = True, as_tensor: bool = True):

        ids = [
            ([self._tokens_to_ids[self.cls_token]] + \
            [self._tokens_to_ids.get(token, self._tokens_to_ids[self.unk_token]) for token in sample[:self.max_seq_len-2]] + \
            [self._tokens_to_ids[self.sep_token]] + \
            [self._tokens_to_ids[self.pad_token]] * ((self.max_seq_len-len(sample)-2) * int(pad)))
            for sample in text]
        
        return torch.tensor(ids, requires_grad=False) if as_tensor else ids
    
    @property
    def vocab_size(self):
        return len(self._tokens_to_ids)
    
    @property
    def pad_token_idx(self):
        return self._tokens_to_ids[self.pad_token]
    
    @property
    def unk_token_idx(self):
        return self._tokens_to_ids[self.unk_token]
    
    @property
    def mask_token_idx(self):
        return self._tokens_to_ids[self.mask_token]
    
    @property
    def cls_token_idx(self):
        return self._tokens_to_ids[self.cls_token]
    
    @property
    def sep_token_idx(self):
        return self._tokens_to_ids[self.sep_token]
    
def get_tokenizer(tokenizer_config: dict):
    
    if tokenizer_config.get('pretrained'):
        tokenizer = CharacterTokenzier(
            chars=tokenizer_config['chars'],
            max_seq_len=tokenizer_config['max_seq_len'],
            **tokenizer_config['special_tokens']
            )
    return tokenizer