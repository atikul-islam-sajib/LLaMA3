import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("./src/")

from model import LLaMA3


class Trainer:
    def __init__(
        self,
        dataloader: DataLoader = None,
        epochs: int = 100,
        lr=1e-5,
        beta1=0.9,
        beta2=0.999,
        device: str = "cpu",
    ):
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.model = LLaMA3(
            dimension=512,
            num_vocabularies=4096,
            query_heads=8,
            num_layers=1,
            kv_heads=4,
            eps=1e-4,
            sequence_length=128,
            base=10000,
            output_dimension=14336,
        )

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimzer = optim.Adam(
            params=self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

    def train(self):
        for epoch in range(self.epochs):
            train_loss = []
            for texts, labels in self.dataloader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(texts)

                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )
                train_loss.append(loss.item())

                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {sum(train_loss)/len(train_loss)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer class for LLaMA3".title())
    parser.add_argument(
        "--dataloader", type=DataLoader, default=None, help="Dataloader".title()
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".title()
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate".title())
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer".title()
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam optimizer".title()
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use".title()
    )

    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    device = args.device

    texts = torch.randint(0, 4096, (64, 128))
    labels = torch.randint(0, 4096, (64, 128))

    dataloader = DataLoader(
        dataset=list(zip(texts, labels)), batch_size=64, shuffle=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trainer = Trainer(
        dataloader=dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    trainer.train()
