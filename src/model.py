import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.constants import (
    DATA_FOLDER,
    MODEL_FOLDER,
    INPUT_SIZE,
    OUTPUT_SIZE,
    LAYERS,
    EPOCHS,
    PREDICTION_THRESHOLD,
)
from src.utils import hamming_score


def mlp(
    input_size: int,
    layer_sizes: list,
    output_size: int,
    output_activation=torch.nn.Softmax,
    activation=torch.nn.LeakyReLU,
):
    input_size = [*input_size] if isinstance(input_size, tuple) else [input_size]
    sizes = input_size + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
        if i < len(sizes) - 2:
            layers += [torch.nn.BatchNorm1d(sizes[i + 1])]
    return torch.nn.Sequential(*layers)


class Network(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        layer_sizes: list,
        output_size: int,
    ) -> None:
        super(Network, self).__init__()

        self.model = torch.nn.DataParallel(
            mlp(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)
        )

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        layer_sizes: list = LAYERS,
        output_size: int = OUTPUT_SIZE,
    ) -> None:

        np.random.seed(42)
        torch.manual_seed(42)

        self.model = Network(input_size, layer_sizes, output_size)
        self.model.to(torch.device("cuda"))

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on the GPU.\n")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.005,
            weight_decay=1e-4,
        )

        self.epoch = 1

        self.model_path = os.path.join(MODEL_FOLDER, "model.pt")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.loss = checkpoint["loss"]

        self.model.train()

    @staticmethod
    def loss_function(prediction, target):
        return torch.nn.functional.binary_cross_entropy(prediction, target)

    def fit(self, train_loader):

        device = next(self.model.parameters()).device
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float()
            self.optimizer.zero_grad()
            prediction = self.model(data)
            loss = self.loss_function(prediction, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        self.epoch += 1

    def test(self, test_loader, threshold=PREDICTION_THRESHOLD):
        self.model.eval()
        self.test_loss = 0
        device = next(self.model.parameters()).device
        with torch.no_grad():
            preds, targs = [], []
            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).float()
                output = self.model(data)
                self.test_loss += self.loss_function(output, target)
                preds.append(np.where(output.cpu() > threshold, 1.0, 0.0))
                targs.append(target.cpu())

        self.test_loss /= len(test_loader.dataset)
        accuracy = hamming_score(np.array(targs), np.array(preds))
        print(f"Average loss: {self.test_loss:.4f}, Accuracy score: {accuracy:.0f}%\n")

    def save_checkpoint(self):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.test_loss,
            },
            self.model_path,
        )


def load_data(directory: str = DATA_FOLDER) -> tuple:
    with open(os.path.join(DATA_FOLDER, "X.npy"), "rb") as f:
        X = np.load(f)
    with open(os.path.join(DATA_FOLDER, "y.npy"), "rb") as f:
        y = np.load(f)

    return X, y


if __name__ == "__main__":

    trainer = Trainer()
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    train_dataloader = DataLoader(
        list(zip(X_train, y_train)), batch_size=64, shuffle=True
    )
    test_dataloader = DataLoader(list(zip(X_test, y_test)), batch_size=64, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        trainer.fit(train_dataloader)
        trainer.test(test_dataloader)
        trainer.save_checkpoint()