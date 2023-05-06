import torch
from .modelbase import ModelBase
class User(ModelBase):
    """
    A class for evaluating and using a model.

    Parameters
    ----------
    model: torch.nn.Module
        The user model to evaluate.
    n_classes: int
        The number of classes in the dataset.
    dataset: torch.utils.data.DataLoader
        The test dataset to evaluate the model on.
    device: torch.device, optional
        The device to use for evaluation (default: 'cuda:0' if available, otherwise 'cpu').

    Examples
    --------
    >>> model = torch.load("model_medmnist/fifth_model.mdl")
    >>> use_model = User(model, 9, test_loader, device)
    >>> use_model.evaluate()
    """
    def __init__(self, model, n_classes, dataset, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)
        self.n_classes = n_classes
        self.test_loader = dataset
        self.device = device

    def evaluate(self):
        """
        Evaluate the user model on the test dataset and calculate evaluation metrics.
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
        self.evaluate_result()