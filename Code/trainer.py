import torch
import time 
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader



class Trainer:
    def __init__(self, TrainConfig, model, train_data):
        self.config = TrainConfig
        self.model = model
        self.data = train_data
        self.optimizer = None
        self.callbacks = defaultdict(list)
        # determine the device we'll train on

        if TrainConfig.device == 'BigMac':
            self.device = "mps" #torch.device("mps")           
        elif TrainConfig.device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)

        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader, expects x, y pairs in dataset
        train_loader = DataLoader(
            self.data,
            sampler=torch.utils.data.RandomSampler(self.data, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        # writer = SummaryWriter()

        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            _, self.loss = model.forward(x, y)
            
            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

            # if self.iter_num % 5 == 0:
            #     print(f'Current epoch: {self.iter_num}, current train loss:{self.loss}')

        # writer.flush()