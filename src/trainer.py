import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("./src/")

from model import LLaMA3
from utils import config_files, device_init


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
        
        self.device = device_init(device=self.device)

        self.model = LLaMA3(
            dimension=config_files()["LLaMA"]["dimension"],
            num_vocabularies=config_files()["LLaMA"]["num_vocabularies"],
            query_heads=config_files()["LLaMA"]["query_heads"],
            num_layers=config_files()["LLaMA"]["num_layers"],
            kv_heads=config_files()["LLaMA"]["kv_heads"],
            eps=float(config_files()["LLaMA"]["eps"]),
            sequence_length=config_files()["LLaMA"]["sequence_length"],
            base=config_files()["LLaMA"]["base"],
            output_dimension=config_files()["LLaMA"]["output_dimension"],
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

    trainer = Trainer(
        dataloader=dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    trainer.train()
