import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split  # type: ignore
from typing import Any

from src.config import Config
from src.constants import DATA_FOLDER, MODEL_FOLDER
from src.utils import hamming_score


def mlp(
    input_size: Any,
    layer_sizes: list,
    output_size: int,
    output_activation=torch.nn.Softmax,
    activation=torch.nn.LeakyReLU,
) -> torch.nn.Sequential:
    input_size = input_size[0] if isinstance(input_size, tuple) else input_size

    layers = [torch.nn.Linear(input_size, layer_sizes[0]), activation()]
    for i in range(len(layer_sizes) - 1):
        layers += [
            torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            activation(),
            torch.nn.BatchNorm1d(layer_sizes[i + 1]),
        ]
    layers += [torch.nn.Linear(layer_sizes[-1], output_size), output_activation(dim=1)]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Trainer:
    def __init__(
        self,
        input_size: int,
        layer_sizes: list,
        output_size: int,
        num_epochs: int,
        learning_rate: float,
        gamma: float,
        decision_threshold: float,
        seed: int = 42,
        preload: bool = True,
    ) -> None:

        self.epoch = 1
        self.num_epochs = num_epochs
        self.decision_threshold = decision_threshold

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = Network(input_size, layer_sizes, output_size)
        self.model.to(torch.device("cuda"))

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on the GPU.\n")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=1,
            gamma=gamma,
        )

        self.model_path = os.path.join(MODEL_FOLDER, "model.pt")
        if preload:
            self.load_checkpoint()

        self.model.train()

    def loss_function(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy(prediction, target)

    def fit(self, train_loader: DataLoader) -> None:
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
                        len(train_loader.dataset),  # type: ignore
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        while self.epoch < self.num_epochs:
            self.fit(train_loader)
            self.test(test_loader)
            self.save_checkpoint()
            self.epoch += 1
            self.scheduler.step()

    def test(self, test_loader: DataLoader) -> None:
        self.model.eval()
        device = next(self.model.parameters()).device
        self.test_loss = torch.zeros_like(torch.empty(1)).to(device)
        with torch.no_grad():
            preds, targs = [], []
            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).float()
                output = self.model(data)
                self.test_loss += self.loss_function(output, target)
                preds.append(np.where(output.cpu() > self.decision_threshold, 1.0, 0.0))
                targs.append(target.cpu())

        self.test_loss /= float(len(test_loader.dataset))  # type: ignore
        accuracy = hamming_score(np.array(targs), np.array(preds))
        loss_str = str(round(self.test_loss.cpu().numpy()[0], 3))
        print(f"Average loss: {loss_str}, Accuracy score: {accuracy:.0f}%\n")

    def save_checkpoint(self) -> None:
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.test_loss,
            },
            self.model_path,
        )

    def load_checkpoint(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("no pre-trained model found in the models folder")

        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = np.array([x]) if x.ndim == 1 else x
            output = self.model(torch.from_numpy(x))
            return np.where(output.cpu() > self.decision_threshold, 1.0, 0.0)[0]


def load_data(directory: str = DATA_FOLDER) -> tuple:
    if not os.listdir(directory):
        raise FileNotFoundError("Pre-processed data is missing.")

    with open(os.path.join(directory, "X.npy"), "rb") as f:
        X = np.load(f)
    with open(os.path.join(directory, "y.npy"), "rb") as f:
        y = np.load(f)

    return X, y


def train(config: Config) -> None:
    trainer = Trainer(*config.model_params())
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=config.seed)
    train_dataloader = DataLoader(
        list(zip(X_train, y_train)), batch_size=config.batch_size, shuffle=True  # type: ignore
    )  # type: ignore
    test_dataloader = DataLoader(list(zip(X_test, y_test)), batch_size=config.batch_size, shuffle=True)  # type: ignore

    trainer.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
