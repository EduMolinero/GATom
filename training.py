
import numpy as np
import pandas as pd

import sys
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score


class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def normalize(self, data):
        data = (data - self.mean) / self.std
        return data
    
    def denormalize(self, data):
        data = (data * self.std) + self.mean
        return data
    
    def __repr__(self):
        return f"Mean: {self.mean}, Std: {self.std}"

class EarlyStopper:
    def __init__(
            self,
            patience : int = 1,
            min_delta: float = 0.0,
            cumulative_delta: bool = False
            ):
        
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Trainer:
    __slots__ = [
        'device', 'epochs', 'save_every',
        'task','loss_function', 'parallel_bool', 
        'rank', 'eval_metric',
    ]
    def __init__(self, params: dict) -> None:
        for key, value in params.items():
            if key in self.__slots__:
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid attribute of {self.__class__.__name__}")
            
    def __repr__(self) -> None:
        print("-"*50)
        print(f"{self.__class__.__name__} with the following attributes:")
        for key in self.__slots__:
            print(f"{key}: {getattr(self, key)}")
        print("-"*50)


    def training(self, loader, model, optimizer):
        total_loss = 0
        norm = 0
        model.train()
        for data in loader:
            data = data.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = getattr(F,self.loss_function)(out, data.y)
            loss.backward()
            total_loss += loss.item() * out.size(0)
            optimizer.step()
            norm += out.size(0)

        total_loss /= norm
        return total_loss 
    
    def evaluation(self, loader, model):
        total_error = 0
        norm = 0
        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device, non_blocking=True)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

                if self.eval_metric != 'auc':
                    error = getattr(F, self.eval_metric)(out, data.y)
                    total_error += error.item() * out.size(0)
                    norm += out.size(0)
                else:
                    preds.extend(out.sigmoid().cpu().numpy())
                    targets.extend(data.y.cpu().numpy())

        if self.eval_metric != 'auc':
            total_error /= norm
        else:
            total_error = roc_auc_score(targets, preds)

        return total_error
    
    def save_checkpoint(self, model, epoch, path):
        ckp = model.module.state_dict() if self.parallel_bool else model.state_dict()
        PATH = os.path.join(path,"checkpoint.pt")
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(ckp, PATH)
        if self.rank == 0:
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_weight_decay(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['weight_decay']

def bytes_to(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    return bytes / (bsize ** a[to])


def train_model(
        model,
        optimizer, 
        train_loader : DataLoader,
        validation_loader : DataLoader,
        test_loader : DataLoader,
        trainer : Trainer,
        scheduler = None,
        normalizer : Normalizer = None,
        early_stopping : bool = True,
        bool_plot : bool = False,
        distributed : bool = False,
        ):

    if trainer.rank == 0:
        print("Model characteristics:")
        print(model)
        print("Training the model.....")
        sys.stdout.flush()
    start = time.time()
    mean_time = 0
    best_val_error = best_test_error = best_epoch = None
    t = []
    loss_epoch = []
    val_epoch = []
    test_epoch = []

    early_stopper = EarlyStopper(patience=50, min_delta=1e-4) 

    for epoch in range(1, trainer.epochs + 1):
        if epoch == 1 and trainer.rank == 0:
            print("Starting training...")
            print("Remainder: All losses/errors are given in standirized units [ std. units ]")
            sys.stdout.flush()
            

        start_epoch = time.time() 
        loss = trainer.training(train_loader, model, optimizer)

        if trainer.rank == 0:
            val_metric = trainer.evaluation(validation_loader, model)
            test_metric = trainer.evaluation(test_loader, model)
            if trainer.task == 'regression':
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Val MAE: {val_metric:.4f} '
                    f'Test MAE: {test_metric:.4f}')
                print(f"Time for epoch {epoch}: {time.time() - start_epoch:.4f} seconds")
                sys.stdout.flush()
                if best_val_error is None or val_metric <= best_val_error:
                    best_val_error = val_metric
                    best_test_error = test_metric
                    best_epoch = epoch
            else:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Val ROCAUC: {val_metric:.4f} '
                    f'Test ROCAUC: {test_metric:.4f}')
                print(f"Time for epoch {epoch}: {time.time() - start_epoch:.4f} seconds")
                sys.stdout.flush()
                if best_val_error is None or val_metric >= best_val_error:
                    best_val_error = val_metric
                    best_test_error = test_metric
                    best_epoch = epoch

            mean_time += time.time() - start_epoch
            t.append(epoch)
            loss_epoch.append(loss)
            val_epoch.append(val_metric)
            test_epoch.append(test_metric)

            if epoch % trainer.save_every == 0:
                trainer.save_checkpoint(model, epoch, 'checkpoints')

            # early stopping
            if early_stopper.early_stop(val_metric) and early_stopping:
                print(f"        --> Early stopping at epoch {epoch} <--")
                sys.stdout.flush()
                break
            

        if distributed:
            torch.distributed.barrier()

        # scheduler
        if scheduler is not None: 
            scheduler.step(val_metric)


    cuda_max_mem = bytes_to(torch.cuda.max_memory_allocated(), 'g')
    if trainer.rank == 0:
        print(f"Total time: {time.time() - start:.4f} seconds || {(time.time() - start)/60 :.4f} minutes || {(time.time() - start)/3600 :.4f} hours")
        print(f"Mean time per epoch: {mean_time/trainer.epochs:.4f} seconds")
        print("Best validation error: ", best_val_error)
        if normalizer is not None:
            print("Best validation error (denormalized): ", normalizer.std.item() * best_val_error)
        print("CUDA memory allocated: ", bytes_to(torch.cuda.memory_allocated(), 'g')," GB")
        print("Max CUDA memory allocated: ", cuda_max_mem, " GB")
        print("Model trained!")
        print('='*50)
        print()
        sys.stdout.flush()
        torch.cuda.empty_cache()

        if bool_plot:
            if trainer.task == 'regression':
                fig, ax = plt.subplots()
                if normalizer is not None:
                    loss_epoch = [x * normalizer.std for x in loss_epoch]
                    val_epoch = [x * normalizer.std for x in val_epoch]
                    test_epoch = [x * normalizer.std for x in test_epoch]
                ax.plot(t, loss_epoch, label='Loss')
                ax.plot(t, val_epoch, label='MAE Validation')
                ax.plot(t, test_epoch, label='MAE Test')
                ax.set(xlabel='Epoch', ylabel='Loss/MAE [Physical units]',
                    title=f'Model performance over {trainer.epochs} epochs')
                ax.grid()
                ax.legend()
                fig.savefig('accuracy.png')
                try:
                    np.savetxt('loss.txt', np.column_stack((t, loss_epoch)))
                    np.savetxt('val.txt', np.column_stack((t, val_epoch)))
                    np.savetxt('test.txt', np.column_stack((t, test_epoch)))
                except:
                    print("Error saving the files")
                    sys.stdout.flush()
            else:
                fig, axs = plt.subplots(1,2)
                axs[0].plot(t, loss_epoch, label='Cross-entropy')
                axs[0].set(xlabel='Epoch', ylabel='Loss')
                
                axs[1].plot(t, val_epoch, label='Validation')
                axs[1].plot(t, test_epoch, label='Test')
                axs[1].set(xlabel='Epoch', ylabel='AUC')

                for i in range(2):
                    axs[i].grid()
                    axs[i].legend()
                fig.savefig('accuracy.png')
                try:
                    np.savetxt('loss.txt', np.column_stack((t, loss_epoch)))
                    np.savetxt('val.txt', np.column_stack((t, val_epoch)))
                    np.savetxt('test.txt', np.column_stack((t, test_epoch)))
                except:
                    print("Error saving the files")
                    sys.stdout.flush()

        # Only works for GATOM models
        dict_model = model._dict_model() if not trainer.parallel_bool else model.module._dict_model()
        dict_model.update({
                'best_val_error': best_val_error,
                'best_test_error': best_test_error,
                'best_epoch': best_epoch,
                'cuda_max_mem': cuda_max_mem,
                'lr': get_lr(optimizer),
                'weight_decay': get_weight_decay(optimizer), 
                'date' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        pd.DataFrame(dict_model, index=[0]).to_csv('results.csv', mode='a')

    return best_val_error, best_test_error