import torch
from .modelbase import ModelBase
from torch.utils.data import DataLoader


class ModelFuser(ModelBase):
    """
    Class for fusing the predictions of multiple models.

    Parameters
    ----------
    *models : ModelBase objects
        Multiple instances of ModelBase objects to be fused.
    test_loader : torch.utils.data.DataLoader
        DataLoader object for test dataset.
    device : torch.device, optional
        Device to use for computation. Default is 'cuda:0' if
        torch.cuda.is_available() is True, otherwise 'cpu'.

    Attributes
    ----------
    models : list
        List of ModelBase objects.
    n_models : int
        Number of models to fuse.
    test_loader : torch.utils.data.DataLoader
        DataLoader object for test dataset.
    n_classes : int
        Number of classes.
    device : torch.device
        Device used for computation.
    target : numpy.ndarray
        Target values of test dataset.
    fused_prediction : numpy.ndarray
        Fused predictions of all models for test dataset.
    evaluation_target : numpy.ndarray
        Target values used for evaluation.
    evaluation_prediction : numpy.ndarray
        Fused predictions used for evaluation.
    probability : numpy.ndarray
        Fused probabilities for all classes.



    Examples
    --------
    >>> model_a = ModelA()
    >>> model_b = ModelB()
    >>> dataset = MyDataset('test')
    >>> loader = DataLoader(dataset, batch_size=32)
    >>> fuser = ModelFuser(model_a, model_b, test_loader=loader)
    >>> fuser.fuse()
    >>> fuser.evaluate_result()
    >>> fuser.plot_evaluate_result()

    """
    def __init__(self, *models, test_loader=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """
        Initializes the ModelFuser object.

        Parameters
        ----------
        *models : ModelBase objects
            Multiple instances of ModelBase objects to be fused.
        test_loader : torch.utils.data.DataLoader
            DataLoader object for test dataset.
        device : torch.device, optional
            Device to use for computation. Default is 'cuda:0' if
            torch.cuda.is_available() is True, otherwise 'cpu'.
        """

        self.models = models
        self.n_models = len(models)
        self.test_loader = test_loader
        self.n_classes = 9
        self.device = device
        assert isinstance(test_loader, torch.utils.data.DataLoader), "Loader is not of type DataLoader"

    def fuse(self):
        """
        Fuses the predictions of all models.
        """
        import torch
        import numpy as np
        assert self.test_loader, "Kein Testdatensatz!"

        predictions = []
        targets = []
        for model in self.models:
            prediction = []
            model.to(self.device)
            model.eval()
            predictions_per_model = []
            targets_per_model = []
            with torch.no_grad():
                for batch, (feature, target) in enumerate(self.test_loader):
                    feature, target = feature.to(self.device), target.to(self.device)
                    prediction = model(feature)
                    targets_per_model.append(target)
                    predictions_per_model.append(prediction)
            predictions.append(torch.stack(predictions_per_model, dim=0))
            targets.append(torch.stack(targets_per_model, dim=0))
        predictions = torch.stack(predictions, dim=0)
        target = torch.stack(targets, dim=0)
        probabilities = torch.nn.functional.softmax(predictions, dim=3).detach().cpu().numpy()
        combined_probability = np.apply_along_axis(lambda row: self.combine_probabilities(row), axis=1, arr=probabilities.reshape([-1, self.n_models * self.n_classes]))
        all_pred = predictions.detach().cpu().numpy().argmax(3).reshape((self.n_models, -1))
        all_target = target[0].detach().cpu().numpy().reshape(-1)
        data = np.vstack([all_target, all_pred]).T
        probabilities = probabilities.reshape((self.n_models, -1, self.n_classes))
        for i in probabilities:
            data = np.hstack([data, i])
        fused_prediction = np.apply_along_axis(lambda row: self.apply_voting(row), axis=1, arr=data)
        result = np.vstack([all_target, fused_prediction])
        self.target = all_target.reshape(-1)
        self.fused_prediction = fused_prediction.reshape(-1)
        self.evaluation_target = self.target
        self.evaluation_prediction = self.fused_prediction
        self.probability = combined_probability

    def combine_probabilities(self, row):
        """
        Computes the combined probability of all models for each class.

        Parameters
        ----------
        row : numpy.ndarray
            Row of fused probabilities for all models.

        Returns
        -------
        numpy.ndarray
            Combined probability for each class.
        """
        import numpy as np
        probabilities = row.reshape([self.n_models, self.n_classes])
        combined_probability = np.mean(probabilities, axis=0)
        return combined_probability

    def apply_voting(self, row):
        """
        Applies voting mechanism to select final prediction.

        Parameters
        ----------
        row : numpy.ndarray
            Row of fused predictions and probabilities for all models.

        Returns
        -------
        int
            Final prediction selected by voting mechanism.
        """
        import numpy as np
        import math
        from scipy.special import softmax
        vote_array = row[1:self.n_models+1].astype(int)
        total_votes = self.n_models
        votes_for_results = math.ceil(total_votes/2)
        max_val = np.argmax(np.bincount(vote_array))
        unique_vals, counts = np.unique(vote_array, return_counts=True)

        if counts.shape[0] <= votes_for_results:
            voted_value = max_val
            return voted_value
        else:
            range_probabilities = self.n_classes * self.n_models
            probabilities = row[-range_probabilities:].reshape([self.n_models, self.n_classes])
            probabilities = softmax(probabilities, axis=1)
            combined_probability = np.mean(probabilities, axis=0)
            voted_value = np.argmax(combined_probability)
            return voted_value
