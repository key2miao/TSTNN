import torch
import os
import shutil
import numpy as np


class Checkpoint(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_val_loss=float("inf"),
                 prev_val_loss=float("inf"), state_dict=None, optimizer=None, num_no_improv=0, half_lr=False):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss

        self.best_val_loss = best_val_loss
        self.prev_val_loss = prev_val_loss

        self.state_dict = state_dict
        self.optimizer = optimizer

        self.num_no_improv = num_no_improv
        self.half_lr = half_lr


    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')


    def load(self, filename):
        # filename : model path
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')

            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss

            self.best_val_loss = checkpoint.best_val_loss
            self.prev_val_loss = checkpoint.prev_val_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
            self.num_no_improv = checkpoint.num_no_improv
            self.half_lr = checkpoint.half_lr
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)