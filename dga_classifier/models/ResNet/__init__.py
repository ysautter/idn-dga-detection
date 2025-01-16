"""
Copyright (C) 2025  <name of author>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""

from dga_classifier import trainer, get_settings, get_name
from dga_classifier.data.dataset import label_to_int
from dga_classifier.models.ResNet.model import ResNet

settings = get_settings()


def get_resnet_model(cfg):
    model_config = cfg.get("model_config")

    model = ResNet(**model_config)
    return model


def train(train_dataset, eval_dataset):
    if settings.classification_task == "binary":
        num_labels = 2
    else:
        num_labels = len(label_to_int)
    return trainer.train(
        name=get_name(),
        get_model=get_resnet_model,
        hparams={
            "model_config": {
                "vocab_size": 40,
                "num_labels": num_labels,
            },
            "batch_size": 128,
            "learning_rate": 5e-4,
            "early_stopping_grace": 5,
            "num_epochs": 20,
        },
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
