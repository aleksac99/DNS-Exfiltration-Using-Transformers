import argparse
import json
import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from exfiltration.config_utils import *
from exfiltration.loggers import DummyLogger, WandBLogger
from exfiltration.preprocess import get_preprocess_fns
from exfiltration.trainer import Trainer


def parse_args():
     
    parser = argparse.ArgumentParser(
         "Run DNS Exfiltration model training."
    )
    parser.add_argument("config_dir", type=str, help="Path to config file.")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    with open(args.config_dir, 'r') as f:
        run_config = json.load(f)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_config, tokenizer_config = run_config['model'], run_config['tokenizer']

    # Select task
    ModelClass, DatasetClass = get_model_data_classes_from_task(run_config['task'])

    # Create Tokenizer
    if run_config['pretrained'] or tokenizer_config.get('pretrained'):

        tokenizer = CharacterTokenzier.from_pretrained(tokenizer_config['load_path'])
        tokenizer.to_config(tokenizer_config)
    else:

            tokenizer = CharacterTokenzier(
            chars=run_config['tokenizer']['chars'],
            max_seq_len=run_config['tokenizer']['max_seq_len'],
            **run_config['tokenizer']['special_tokens'])

            tokenizer.to_file(os.path.join(tokenizer_config['save_path']))

    # Create Model
    model = ModelClass(
        out_features=model_config.get('out_features'),
        n_layers=model_config.get('n_layers'),
        vocab_size=tokenizer.vocab_size,
        max_seq_len=model_config.get('max_seq_len'),
        d_model=model_config.get('d_model'),
        n_heads=model_config.get('n_heads'),
        d_feedforward=model_config.get('d_feedforward'),
        padding_idx=tokenizer.pad_token_idx,
        dropout_prob=model_config.get('dropout_prob', 0.1))

    if run_config['pretrained']:
        model.load_weights(run_config['model'], tokenizer)

    # Create Datasets and DataLoaders
    train_fns = get_preprocess_fns(
        [[fn['name'] for fn in element] for element in run_config['data']['train']['preprocess']])
    train_params = [[fn.get('params', {}) for fn in element] for element in run_config['data']['train']['preprocess']]

    train_dataset = DatasetClass(
         paths=run_config['data']['train']['paths'],
         tokenizer=tokenizer,
         preprocess_fns=train_fns,
         preprocess_params=train_params,
         ignore_index=-100,
         **run_config['data']['train'].get('other_params', dict()))

    eval_datasets = {}
    for dataset in run_config['data']['eval']:
        
        eval_fns = get_preprocess_fns(
            [[fn['name'] for fn in element] for element in dataset['info']['preprocess']])
        eval_params = [[fn.get('params', {}) for fn in element] for element in dataset['info']['preprocess']]

        eval_datasets[dataset['name']] = DatasetClass(
            paths=dataset['info']['paths'],
            tokenizer=tokenizer,
            preprocess_fns=eval_fns,
            preprocess_params=eval_params,
            ignore_index=-100,
            **run_config['data']['train'].get('other_params', dict()))
    
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Eval datasets length: {[len(d) for d in eval_datasets]}')

    # 4. Logging
    run_config['training']['device'] = device
    run_config['data']['train']['n_samples'] = len(train_dataset)
    run_config['model']['n_parameters'] = sum(
        [np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    
    if run_config['task']=='finetuning':
        run_config['data']['train']['n_positives'] = train_dataset.labels.sum()

    logger = WandBLogger(**run_config['logger']['params'],
                         config=run_config, group=run_config['task'].upper()) if run_config['logger']['name'] == 'WandB' else DummyLogger()

    # 5. Training
    criterion = CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(),
                      run_config['training']['learning_rate'])

    trainer = Trainer(model, optimizer, criterion, train_dataset,
                      eval_datasets, logger, run_config['training']['batch_size'],
                      run_config['training']['n_epochs'], run_config['training']['evaluate_each'],
                      run_config['training']['evaluate_on_epoch_end'], run_config['training']['metrics'],
                      run_config['training']['save'], device)

    trainer.train()

    print('Training completed successfully.')
