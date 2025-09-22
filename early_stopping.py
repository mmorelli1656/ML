# -*- coding: utf-8 -*-
"""
Early stopping for Deep Learning algorithms with secure checkpoints.
Author: mik16
"""

import torch
import numpy as np


class EarlyStopping:
    """
    Handles early stopping during model training.

    Args:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as an improvement.
        save_path (str): Path where the best model will be saved.
        config (dict, optional): Extra configuration to be saved with the model.
        verbose (bool): If True, prints messages when the model is saved or early stopping is triggered.

    Example:
        >>> early_stopping = EarlyStopping(patience=5, min_delta=0.01, save_path='checkpoint.pth', verbose=True)
        >>> for epoch in range(epochs):
        >>>     train(...)  # your training code
        >>>     val_loss = evaluate(...)  # your validation loss computation
        >>>     early_stopping(val_loss, model)
        >>>     if early_stopping.early_stop:
        >>>         break
    """

    def __init__(self, patience, min_delta, save_path, config=None, verbose=False):
        # How many epochs to wait without improvement before stopping
        self.patience = patience
        
        # Minimum improvement in validation loss to reset the counter
        self.min_delta = min_delta
        
        # Path to save the best model checkpoint
        self.save_path = save_path
        
        # Whether to print progress messages
        self.verbose = verbose
        
        # Optional dictionary to save additional configuration
        self.config = config or {}

        # Initialize best loss as infinity so any real loss will be smaller
        self.best_loss = np.inf
        
        # Counter for epochs without improvement
        self.counter = 0
        
        # Flag to indicate whether early stopping should trigger
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Call this method at the end of each epoch with the current validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): The model being trained.
        """
        # Check if the current validation loss is significantly better than the best so far
        if val_loss < self.best_loss - self.min_delta:
            # New best loss found — reset counter
            self.best_loss = val_loss
            self.counter = 0

            # Prepare checkpoint data: model weights + validation loss + config
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }

            # Save checkpoint to file
            torch.save(checkpoint, self.save_path)

            if self.verbose:
                print(f"Validation loss improved. Saving model and config to {self.save_path}")
        else:
            # No significant improvement — increment the counter
            self.counter += 1

            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epoch(s).")

            # If no improvement has been seen for `patience` epochs, trigger early stopping
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement.")
