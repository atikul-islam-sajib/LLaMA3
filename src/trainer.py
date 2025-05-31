import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
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

                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()

                train_loss.append(loss.item())

            print(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {sum(train_loss)/len(train_loss)}"
            )


if __name__ == "__main__":
    texts = torch.randint(0, 4096, (64, 128))
    labels = torch.randint(0, 4096, (64, 128))

    dataloader = DataLoader(
        dataset=list(zip(texts, labels)), batch_size=64, shuffle=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trainer = Trainer(dataloader=dataloader, device=device, epochs=2)

    trainer.train()
