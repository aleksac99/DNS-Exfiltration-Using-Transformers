from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from exfiltration.my_tokenizers import CharacterTokenzier

class Embeddings(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int,
                 padding_idx: int,
                 dropout_prob: int,
                 *args, **kwargs) -> None:

        super().__init__()

        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx)
        
        self.positional_embeddings = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model)
        
        self.token_type_embeddings = nn.Embedding(
            num_embeddings=2, # NOTE: Token type embeddings aren't used, but are included for compatibility reasons
            embedding_dim=d_model)

        self.ln = nn.LayerNorm((d_model, ))
        self.dropout = nn.Dropout(dropout_prob)

        positions = torch.arange(max_seq_len)
        self.register_buffer('positions', positions)

    def forward(self, x, **kwargs):
        
        seq_len = x.shape[1]
        positions = self.positions[:seq_len]
        token_type_ids = kwargs.get('token_type_ids', torch.zeros_like(x))
        x = self.word_embeddings(x) + self.positional_embeddings(positions) + self.token_type_embeddings(token_type_ids)
        x = self.ln(x)
        x = self.dropout(x)

        return x

class Bert(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 n_heads: int,
                 d_feedforward: int,
                 n_layers: int,
                 padding_idx: int,
                 dropout_prob: int=0.1,
                 *args, **kwargs) -> None:

        super().__init__()

        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            padding_idx=padding_idx,
            dropout_prob=dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_prob,
            activation='gelu',
            batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,)

    def forward(self, x, **kwargs):

        x = self.embeddings(x)
        x = self.encoder(x)

        return x
    
class BertClassifier(nn.Module):

    def __init__(self,
                 out_features: int,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 n_heads: int,
                 d_feedforward: int,
                 n_layers: int,
                 padding_idx: int,
                 dropout_prob: int=0.1,
                 *args, **kwargs) -> None:

        super().__init__()

        self.bert = Bert(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_feedforward=d_feedforward,
            n_layers=n_layers,
            padding_idx=padding_idx,
            dropout_prob=dropout_prob)
        
        self.cls_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh())
        
        self.dropout = nn.Dropout(dropout_prob)
        self.clf = nn.Linear(d_model, out_features)

    def forward(self, x):

        x = self.bert(x)
        x = x[:, 0] # extract CLS token values
        x = self.cls_pooler(x)
        x = self.dropout(x)
        x = self.clf(x)

        return x
    
    def load_weights(self, model_config: dict, tokenizer: CharacterTokenzier) -> None:

        try:    
            not_found = self.load_state_dict(torch.load(model_config['load_path']), strict=False)
            if not_found.missing_keys != []:
                print(f'Following keys not loaded: {", ".join(not_found.missing_keys)}')
        except Exception as e:
            raise ValueError("Pretrained weights don't match with model definition parameters.") from e

    
def get_model(ModelClass: type, model_config: dict, tokenizer: CharacterTokenzier) -> nn.Module:

    return ModelClass(
        out_features=model_config['out_features'],
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        d_feedforward=model_config['d_feedforward'],
        padding_idx=tokenizer.pad_token_idx,
        dropout_prob=model_config.get('dropout_prob', 0.1))


class BertMLM(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 n_heads: int,
                 d_feedforward: int,
                 n_layers: int,
                 padding_idx: int,
                 dropout_prob: int=0.1,
                 *args, **kwargs) -> None:
        
        super().__init__()

        self.bert = Bert(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_feedforward=d_feedforward,
            n_layers=n_layers,
            padding_idx=padding_idx,
            dropout_prob=dropout_prob)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm((d_model, )))

        self.out = nn.Linear(d_model, vocab_size, bias=True)
        self.out.weight=self.bert.embeddings.word_embeddings.weight
        
    def forward(self, x):
        x = self.bert(x)
        x = self.mlm_head(x)
        x = self.out(x)
        return x.permute(0, 2, 1)
    
    def load_weights(self, model_config: dict, *args) -> None:

        try:
            self.load_state_dict(torch.load(model_config['load_path']))
        except Exception as e:
            raise ValueError("Pretrained weights don't match with model definition parameters.") from e