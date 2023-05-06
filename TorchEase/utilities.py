import numpy as np
class EarlyStopper:
    """
    Early stopper that stops training if the validation loss does not improve
    for a given number of epochs.

    Parameters
    ----------
        patience (int):
            The number of epochs to wait if the validation loss does not improve before stopping early.
        min_delta (float):
            The minimum change in validation loss to be considered as an improvement.

    Attributes
    ----------
        patience (int):
            The number of epochs to wait if the validation loss does not improve before stopping early.
        min_delta (float):
            The minimum change in validation loss to be considered as an improvement.
        counter (int):
            The number of epochs that the validation loss has not improved.
        min_loss (float):
            The best validation loss so far.

    Examples
    --------
    >>> from TorchEase import EarlyStopper
    >>> early_stop = EarlyStopper(patience=15, min_delta=0.01)
    >>> trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, early_stop=early_stop)
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, validation_loss, verbose=False):
        if validation_loss < self.min_loss - self.min_delta:
            self.counter = 0
            self.min_loss = validation_loss
            if verbose:
                print(f"Early Stop: Counter reset. Min loss is {self.min_loss}.")
        else:
            self.counter += 1
            if verbose:
                print(f"Early Stop: Counter increased to {self.counter}. Min loss is {self.min_loss}. Patience is set to {self.patience}.")
            if self.counter >= self.patience:
                if verbose:
                    print("Early Stop: Patience reached. Stopping early.")
                return True
        return False
