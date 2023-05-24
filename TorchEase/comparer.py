import os.path

import torch
from .modelbase import ModelBase


class Comparer(ModelBase):
    def __init__(self, *models, n_classes, data_loader=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.models = models
        self.n_classes = n_classes
        self.n_models = len(models)
        self.data_loader = data_loader
        self.device = device
        self.total_training_time = 0

    def compare(self):
        model_target = []
        model_prediction = []
        with torch.no_grad():
            for batch, (feature, target) in enumerate(self.data_loader):
                _target = []
                _prediction = []
                for model in self.models:
                    model = model.to(self.device)
                    feature, target = feature.to(self.device), target.to(self.device)
                    prediction = model(feature)
                    _target.append(target)
                    _prediction.append(prediction.argmax(1))
                model_target.append(_target[0])
                model_prediction.append(torch.stack(_prediction, dim=1))

            self.test_target = torch.stack(model_target, dim=0).reshape(-1)
            self.test_prediction = torch.stack(model_prediction, dim=0).reshape(-1, self.n_models)

        # self.evaluation_target = self.test_target.detach().cpu().numpy()
        # self.evaluation_prediction = self.test_prediction.detach().cpu().numpy()

    def evaluate_result(self):
        import pandas as pd
        self.evaluation_target = self.test_target.detach().cpu().numpy()
        self._metrics = []
        self._confusion_matrices = []
        for i in range(self.n_models):
            self.evaluation_prediction = self.test_prediction.detach().cpu().numpy()[:, i]
            # Rufen Sie die evaluate_result-Methode von ModelBase auf
            super().evaluate_result(verbose=False)
            self._metrics.append(self.metrics)
            self._confusion_matrices.append(self.confussion_martix)
        # print(self._metrics)

    def save_result(self, path_to_save="", CM_fname="ConfussionMatrix.png", barplot_fname="CompareBarPlot.png"):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        _accuracy = []
        _weighted_f1_score = []
        _weighted_precision = []
        _weighted_recall = []
        _model_names = []

        _model_name = ""
        for i in range(self.n_models):
            # self._metrics[i]["macro avg"]["f1-score"]
            _accuracy.append(self._metrics[i]["accuracy"])
            _weighted_f1_score.append(self._metrics[i]["weighted avg"]["f1-score"])
            _weighted_precision.append(self._metrics[i]["weighted avg"]["precision"])
            _weighted_recall.append(self._metrics[i]["weighted avg"]["recall"])
            _model_name = self.models[i].__class__.__name__
            if _model_name == self.models[i].__class__.__name__:
                _model_name = _model_name + "_" + str(i)
            _model_names.append(_model_name)

            print(i)
        # pd.DataFrame(_accuracy, columns=_model_names, index=["acc"])
        _columns_names = ["Accuracy", "weighted F1-Score", "weighted Precision", "weighted Recall", "Model"]
        df = pd.DataFrame([_accuracy, _weighted_f1_score, _weighted_precision, _weighted_recall, _model_names],
                          index=_columns_names)
        df_transposed = df.T
        df = pd.melt(df_transposed, id_vars="Model", var_name="Metrics", value_name="Values")
        sns.set()
        g = sns.catplot(x='Model', y='Values', col='Metrics', data=df, kind='bar')
        g.set_xticklabels(rotation=45)
        plt.xticks(rotation=45)
        # sns.set(rc={'figure.figsize': (20, 10)})
        plt.savefig(os.path.join(path_to_save, barplot_fname))
        plt.close()
        for index, cm in enumerate(self._confusion_matrices):
            filename, file_extension = os.path.splitext(CM_fname)
            fname = filename + "_" + _model_names[index] + file_extension
            self.cm = cm
            super().save_result(path_to_save=path_to_save, CM_fname=fname)
        # print(_accuracy)




