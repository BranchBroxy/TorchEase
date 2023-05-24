import unittest
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch import optim

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

class TestModel_1(nn.Module):
    def __init__(self):
        super(TestModel_1, self).__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

class TestModel_2(nn.Module):
    def __init__(self):
        super(TestModel_2, self).__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x


class TestDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [5.0, 6.0, 7.0, 8.0]])
        self.targets = torch.tensor([0, 1, 2, 3, 0, 1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target


class TestTorchEase(unittest.TestCase):
    def test_trainer(self):
        from TorchEase import Trainer, EarlyStopper
        train_loader = DataLoader(TestDataset(), batch_size=2, shuffle=True)
        eval_loader = DataLoader(TestDataset(), batch_size=2, shuffle=False)
        test_loader = DataLoader(TestDataset(), batch_size=2, shuffle=False)

        model = TestModel()  # TestModel als Modell verwenden

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        early_stop = EarlyStopper(patience=15, min_delta=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, n_classes=4,
                          train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader,
                          device=device, early_stop=early_stop, scheduler=scheduler, verbose=False)

        trainer.run(epochs=1)  # Anzahl der Epochen für den Test angeben

        # Weitere Assertions hinzufügen, um das Verhalten des Trainers zu überprüfen
        self.assertIsNotNone(trainer.train_loss)
        self.assertIsNotNone(trainer.eval_loss)
        self.assertIsNotNone(trainer.evaluation_target)
        self.assertIsNotNone(trainer.evaluation_prediction)

    def test_fuser(self):
        from TorchEase import ModelFuser
        # Testen Sie die fuse-Methode
        model_a = TestModel()  # TestModel als Modell verwenden
        model_b = TestModel()
        test_loader = DataLoader(TestDataset(), batch_size=2, shuffle=False)
        self.fuser = ModelFuser(model_a, model_b, n_classes=4, data_loader=test_loader)


        self.fuser.fuse()

        # Überprüfen Sie, ob die Attribute richtig gesetzt wurden
        self.assertEqual(len(self.fuser.models), 2)
        self.assertEqual(self.fuser.n_models, 2)
        self.assertEqual(self.fuser.n_classes, 4)
        self.assertIsNotNone(self.fuser.data_loader)
        self.assertIsNotNone(self.fuser.target)
        self.assertIsNotNone(self.fuser.fused_prediction)
        self.assertIsNotNone(self.fuser.evaluation_target)
        self.assertIsNotNone(self.fuser.evaluation_prediction)
        self.assertIsNotNone(self.fuser.probability)

    def test_comparer(self):
        from TorchEase import Comparer
        import copy
        model_a = TestModel()  # TestModel als Modell verwenden
        model_b = TestModel_1()
        model_c = TestModel_2()
        data_loader = DataLoader(TestDataset(), batch_size=2, shuffle=False)
        self.comparer = Comparer(model_a, model_b, model_c, n_classes=4, data_loader=data_loader)
        self.comparer.compare()
        self.assertIsNotNone(self.comparer.test_target)
        self.assertIsNotNone(self.comparer.test_prediction)
        self.comparer.evaluate_result()
        self.comparer.save_result()
