import os
from datetime import datetime
import wandb

class DummyLogger:
    
    def __init__(self, *args, **kwargs) -> None:
        
        self.name = datetime.strftime(
                datetime.now(), '%Y_%m_%d_%H_%M_%S')

    def watch(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def get_run_name(self):
        return self.name

class WandBLogger(DummyLogger):
    
    def __init__(self, *args, **kwargs) -> None:

        wandb.login(key=os.environ['WANDB_API_KEY'])
        self.run = wandb.init(*args, **kwargs)

    def watch(self, model, criterion, log_freq: int = 100):

        wandb.watch(model, criterion, log="all", log_freq=log_freq)

    def log(self, results, section=None, step=None):

        if section is not None:
            results = {'/'.join([section, k]): v for k, v in results.items()}

        wandb.log(results, step=step)

    def finish(self):

        wandb.finish()

    def get_run_name(self):
        return self.run.name

class TensorBoard(DummyLogger):
    pass