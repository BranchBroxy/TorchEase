import torch
from .modelbase import ModelBase
class Trainer(ModelBase):
    """
    A class for training, evaluating and testing PyTorch models.

    Parameters
    ----------
        model (torch.nn.Module): The PyTorch model to be trained, evaluated and tested.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        loss_fn (torch.nn.Module): The loss function to be used for training and evaluation.
        early_stop (object): The early stopping criteria to stop training early if validation loss stops improving.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to be used for training.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        eval_loader (torch.utils.data.DataLoader): The DataLoader for the evaluation dataset.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the testing dataset.
        device (torch.device): The device to be used for training, evaluation and testing.
        verbose (bool): If True, prints the training and evaluation progress. Default is True.
        train_verbose (bool): If True, prints the training progress. Default is True.
        train_verbose_step (int): The number of steps to print during the training progress. Default is 5.
        eval_verbose (bool): If True, prints the evaluation progress. Default is False.
        n_classes (int): The number of classes for the classification task. Default is 10.



    Methods
    ----------
        epoch (int): The current epoch number.
        epochs (int): The total number of epochs to train the model for.
        train_loss (float): The training loss.
        eval_loss (float): The evaluation loss.
        n_classes (int): The number of classes for the classification task.
        evaluation_target (numpy.ndarray): The target values for the evaluation dataset.
        evaluation_prediction (numpy.ndarray): The predicted values for the evaluation dataset.

    Examples
    --------
    >>> model = Model()
    >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
    >>> loss_fn = nn.CrossEntropyLoss()
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> early_stop = EarlyStopper(patience=15, min_delta=0.01)
    >>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    >>> trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, early_stop=early_stop, scheduler=scheduler,
    ...                   train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader, device=device,
    ...                   verbose=True, eval_verbose=True, n_classes=9)
    >>> trainer.run()
    >>> trainer.evaluate_result()
    >>> trainer.save_model("model.pth")
    """
    def __init__(self, model, optimizer, loss_fn, early_stop=None, scheduler=None,
                 train_loader=None, eval_loader=None, test_loader=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 verbose: bool=True, train_verbose: bool=True, train_verbose_step: int=5, eval_verbose: bool=False, n_classes=10):
        if not train_loader and not eval_loader:
            print("Warning: No training and validation dataloader is provided to the Trainer()")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.early_stop = early_stop
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose
        self.train_verbose = train_verbose
        self.train_verbose_step = train_verbose_step
        self.eval_verbose = eval_verbose

        # Training Parameters
        self.epoch = 0
        self.epochs = 0
        self.train_loss = 0

        # Evaluation Parameters
        self.eval_loss = None
        self.n_classes = n_classes
        self.evaluation_target = None
        self.evaluation_prediction = None

    def run(self, epochs: int = 10):
        self.epochs = epochs
        for self.epoch in range(self.epochs):
            if self.train_verbose and self.verbose:
                print(f"Epoch {self.epoch + 1} of {self.model.__class__.__name__}\n-------------------------------")
            self.train()
            if self.eval_loader:
                self.eval()
                if self.early_stop and self.early_stop.early_stop(validation_loss=self.eval_loss, verbose=self.eval_verbose and self.verbose):
                    break
        self.test()

    def train(self):
        """
        Trains the model on the training dataset.
        """
        self.model.train()
        for batch, (feature, target) in enumerate(self.train_loader):
            feature, target = feature.to(self.device), target.to(self.device)
            prediction = self.model(feature)
            try:
                self.train_loss = self.loss_fn(prediction, target)
            except:
                target = target.squeeze(1)
                self.train_loss = self.loss_fn(prediction, target)
                # self.train_loss = self.loss_fn(prediction.argmax(1).float(), target.float())
            self.optimizer.zero_grad()
            self.train_loss.backward()
            self.optimizer.step()
            if self.train_verbose and (batch + 1) % int(len(self.train_loader)/self.train_verbose_step) == 0 and self.verbose:
                print(f"Epoch [{self.epoch + 1}/{self.epochs}], Step [{batch + 1}/{len(self.train_loader)}], "
                      f"Loss: {self.train_loss.item():.4f}, Learning Rate {self.optimizer.param_groups[0]['lr']}")

    def eval(self):
        """
        Evaluates the model on the evaluation dataset.
        """
        with torch.no_grad():
            self.model.eval()
            for batch, (feature, target) in enumerate(self.eval_loader):
                feature, target = feature.to(self.device), target.to(self.device)
                prediction = self.model(feature)
                try:
                    self.eval_loss = self.loss_fn(prediction, target)
                except:
                    target = target.squeeze(1)
                    self.eval_loss = self.loss_fn(prediction, target)
            self.scheduler.step(self.eval_loss)

    def test(self):
        """
        Tests the model on the testing dataset.
        """
        self.model.eval()
        self.test_target = []
        self.test_prediction = []
        with torch.no_grad():
            self.model.eval()
            for batch, (feature, target) in enumerate(self.test_loader):
                feature, target = feature.to(self.device), target.to(self.device)
                prediction = self.model(feature)
                self.test_target.append(target)
                self.test_prediction.append(prediction)
            self.test_target = torch.stack(self.test_target, dim=0).reshape(-1)
            self.test_prediction = torch.stack(self.test_prediction, dim=0).reshape(-1, self.n_classes)

        self.evaluation_target = self.test_target.detach().cpu().numpy()
        self.evaluation_prediction = self.test_prediction.argmax(1).detach().cpu().numpy()

    def save_model(self, path="model.pth"):
        """
        Saves the PyTorch model to the specified path.
        """
        torch.save(self.model, path)

    def __str__(self):
        """
        Returns a string representation of the model, optimizer, loss function and device used for training.
        """
        model_name = self.model.__class__.__name__
        optimizer_name = self.optimizer.__class__.__name__
        loss_fn_name = self.loss_fn.__class__.__name__
        device_name = self.device
        return f"Name of Model:\t\t{model_name}\n" \
               f"Used Optimizer:\t\t{optimizer_name}\n" \
               f"Used Loss Function:\t{loss_fn_name}\n" \
               f"Used Device:\t\t{device_name}"
