import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from dga_classifier import get_settings

settings = get_settings()


class ResNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        vocab_size: int = 38,
        out_channels: int = 256,
        num_labels: int = 2,
    ):
        super().__init__()

        self.name = "ResNet"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.out_channels = out_channels
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool_p = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer0 = Residual(128, 256, kernel_size=4)
        self.layer1 = Residual(256, 256, kernel_size=3)
        self.layer2 = Residual(256, 256)
        self.layer3 = Residual(256, 256)
        self.layer4 = Residual(256, 256)
        if settings.classification_task == "multiclass":
            self.layer5 = Residual(256, 256)
            self.layer6 = Residual(256, 256)
            self.layer7 = Residual(256, 256)
            self.layer8 = Residual(256, 256)
            self.layer9 = Residual(256, 256)
            self.layer10 = Residual(256, 256)
            self.layer11 = Residual(256, 256)
        self.num_labels = num_labels
        self.ll = nn.Linear(512, self.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, labels=None, *args, **kwargs) -> torch.Tensor:
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)

        x = self.layer0(x)
        x = self.maxpool_p(x)

        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool_p(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        if settings.classification_task == "multiclass":
            x = self.layer5(x)
            x = self.maxpool(x)
            x = self.layer6(x)
            x = self.maxpool(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)
            x = self.layer11(x)

        x = torch.flatten(x, 1, 2)

        logits = self.ll(x)

        # logits = x.flatten()

        # prob = self.sigmoid(logits)

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

            return loss

        return logits


class Residual(nn.Module):
    def __init__(self, input_dim: int = 256, out_channels: int = 256, kernel_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.channel_adjust = nn.Conv1d(
            in_channels=input_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        residual = batch_x

        if self.out_channels != self.input_dim:
            residual = self.channel_adjust(batch_x)

        x = self.conv1(residual)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = residual + x

        x = self.relu(x)

        return x
