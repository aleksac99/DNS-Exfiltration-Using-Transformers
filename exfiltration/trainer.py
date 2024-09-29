import json
import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from .metrics import *


class Trainer:

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 train_dataset,
                 test_datasets,
                 logger,
                 batch_size,
                 n_epochs,
                 evaluate_each,
                 evaluate_on_epoch_end,
                 metrics,
                 save,
                 device) -> None:

        print(f'Device: {device}')
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = loss.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_loaders = {
            dataset_name: DataLoader(dataset, batch_size, shuffle=True)
            for dataset_name, dataset in test_datasets.items()}
        self.n_epochs = n_epochs

        self.scheduler = self.__get_lr_scheduler()

        self.epoch_results = []
        self.step_results = []
        self.epoch_train_loss = []
        self.step_train_loss = []
        self.evaluate_each = evaluate_each
        self.evaluate_on_epoch_end = evaluate_on_epoch_end
        self.save = save
        self.device = device
        self.global_step = 0

        self.logger = logger
        self.metrics = metrics
        self.experiment_start = datetime.strftime(
            datetime.now(), '%Y_%m_%d_%H_%M_%S')
        
        self.save_path = os.path.join(
            self.save['save_dir'], self.logger.get_run_name())

        os.makedirs(self.save_path, exist_ok=False)

    def __get_lr_scheduler(self):

        n_steps = self.n_epochs * len(self.train_loader)

        def get_lr_scheduler_lambda(n_steps):

            return lambda current_step: max(0.0, float(n_steps - current_step) / n_steps)

        return LambdaLR(self.optimizer, get_lr_scheduler_lambda(n_steps), -1)

    def train(self):

        self.best_result = 0.
        self.logger.watch(self.model, self.criterion)
        self.epoch_bar = tqdm(range(self.n_epochs))
        for epoch in self.epoch_bar:

            self.model.train()
            self.train_epoch(epoch)

            if self.evaluate_on_epoch_end:

                results = self.evaluate()
                for s, r in results.items():
                    self.logger.log({**r, 'epoch': epoch+1},
                                    section=s, step=self.global_step)

                self.epoch_results.append(
                    {'epoch': epoch+1, 'results': results})

        torch.save(self.model.state_dict(), os.path.join(
            self.save_path, 'final_model.pt'))

        metrics = {
            'epoch_eval_results': self.epoch_results,
            'step_eval_results': self.step_results,
            'epoch_train_loss': self.epoch_train_loss,
            'step_train_loss': self.step_train_loss}
        
        with open(os.path.join(self.save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

        self.logger.finish()

    def train_epoch(self, epoch):

        running_loss = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):

            self.global_step += 1
            if self.global_step % 1_000 == 0:
                self.epoch_bar.set_description(
                    f'Step {self.global_step}. Batch {batch_idx+1}/{len(self.train_loader)}.')

            batch_loss = self.train_batch(features, labels)
            running_loss += batch_loss

            self.logger.log({
                'loss': batch_loss,
                'learning_rate': self.scheduler.get_last_lr()[0], },
                section='train', step=self.global_step)

            if (self.global_step) % self.evaluate_each == 0:
                step_results = self.evaluate()

                for s, r in step_results.items():
                    self.logger.log(r, section=s, step=self.global_step)

                self.step_results.append(
                    {'results': step_results, 'global_step': self.global_step})

        epoch_loss = running_loss / len(self.train_loader)
        self.epoch_train_loss.append(epoch_loss)
        self.logger.log({'epoch_loss': epoch_loss,
                        'epoch': epoch+1}, section='train', step=self.global_step)

    def train_batch(self, features, labels):

        features, labels = features.to(self.device), labels.to(self.device)

        # Forward pass
        logits = self.model(features)

        # Loss
        loss = self.criterion(logits, labels)
        self.step_train_loss.append(loss.item())

        # Zero grad
        self.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Step
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate(self, dataset_name=None):

        self.model.eval()

        results = {}

        if dataset_name is None:
            epoch_log = self.epoch_bar.desc.split('.')[0]
            for name, loader in self.test_loaders.items():
                self.epoch_bar.set_description(
                    '. '.join([epoch_log, f'Evaluating on {name}...']))
                results[name] = self.__evaluate_dataset(loader)
        else:
            results[dataset_name] = self.__evaluate_dataset(loader)

        current_result = results[self.save['dataset']][self.save['metric']]

        if current_result >= self.best_result:

            self.best_result = current_result
            torch.save(self.model.state_dict(), os.path.join(
                self.save_path, 'best_model.pt'))

        self.model.train()

        return results

    def __evaluate_dataset(self, loader):

        preds = []
        targets = []
        losses = []

        epoch_log = self.epoch_bar.desc
        for eval_idx, (features, labels) in enumerate(loader):

            if eval_idx % 500 == 0:
                self.epoch_bar.set_description(epoch_log + f' {eval_idx}/{len(loader)}')
            features, labels = features.to(self.device), labels.to(self.device)

            with torch.no_grad():

                logits = self.model(features)
                _, idx = torch.max(logits, dim=-1)
                preds += idx.tolist()
                targets += labels.tolist()
                loss = self.criterion(logits, labels)
                losses.append(loss.item())

        preds = torch.tensor(preds)
        targets = torch.tensor(targets)
        loss = torch.tensor(losses).mean()
        results = {'loss': loss.item()}

        for m in self.metrics:
            if m == 'perplexity':
                results[m] = perplexity_score(loss)
            elif m == 'precision':
                results[m] = precision_score(preds, targets)
            elif m == 'recall':
                results[m] = recall_score(preds, targets)
            elif m == 'f1':
                results[m] = f1_score(preds, targets)

        return results
