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

import copy
import logging
import math
import time
import torch
import numpy as np
import collections.abc

from omegaconf import OmegaConf
from torch import nn
from typing import List
from sklearn.model_selection import StratifiedKFold

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


from transformers.modeling_outputs import SequenceClassifierOutput

from dga_classifier.models import (
    ResNet,
    huggingface_models, DeBERTa,
)

from dga_classifier import calculate_metrics, settings, __get_all_idgas, get_name
from dga_classifier.data.dataset import (
    get_data,
    get_datasets,
    get_idn_data,
    get_dataset,
    rw_eval_dgas,
    label_to_int,
)

LOG_EVERY = 500
EVAL_EVERY = 0.5

logger = logging.getLogger("dga_classifier.trainer")

def run(
        model,
        tokenizer,
        tokenization,
        include_idga: List[str] = None,
        evaluate_idgas: bool = False,
        rw_iteration: int = 0,
        single_fold: bool = False,
        balanced_idn_training: bool = False,
):
    # Load Data
    domains, labels = get_data(
        preprocess=False,
        tokenizer=tokenizer,
        pre_train=False,
        include_idga=include_idga,
        rw_iteration=rw_iteration,
        balanced_idn_training=balanced_idn_training,
    )
    domains = np.array(domains)
    labels = np.array(labels)

    n_splits = settings.k_folds

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    evaluation_results = []

    # Start iterating over the train-eval splits
    for i, (train_index, validation_index) in enumerate(skf.split(domains, labels)):
        if settings.single_fold or single_fold and i > 0:
            break
        logger.info(f"{i}-fold cross validation")
        train_domains, train_labels = domains[train_index], labels[train_index]
        eval_domains, eval_labels = (
            domains[validation_index],
            labels[validation_index],
        )

        # Create train and eval set
        train_dataset, eval_dataset = get_datasets(
            train_domains,
            train_labels,
            eval_domains,
            eval_labels,
            tokenization,
            tokenizer,
        )

        orig_eval_dataset = []
        if evaluate_idgas or settings.balanced_idn_analysis:
            orig_eval_domains, orig_eval_labels = zip(
                *[
                    (x, y)
                    for x, y in zip(eval_domains, eval_labels)
                    if y <= 115 and y >= 0
                ]
            )
            orig_eval_domains = list(orig_eval_domains)
            orig_eval_labels = list(orig_eval_labels)
            orig_eval_dataset = get_dataset(
                orig_eval_domains, orig_eval_labels, tokenization, tokenizer
            )

        logger.info(f"Start training {model}")

        trained_model = None
        eval_loss_fn = CrossEntropyLoss()

        match model:
            case "Encoder":
                trained_model = huggingface_models.train_model(
                    train_dataset,
                    eval_dataset,
                    tokenizer,
                    pretrained_model_name_or_path="",
                )

            case "BERT":
                trained_model = huggingface_models.train_model(
                    train_dataset,
                    eval_dataset,
                    tokenizer,
                    pretrained_model_name_or_path="bert-base-uncased",
                )

            case "XLM-Roberta":
                trained_model = huggingface_models.train_model(
                    train_dataset,
                    eval_dataset,
                    tokenizer,
                    pretrained_model_name_or_path="FacebookAI/xlm-roberta-base",
                )
            case "DeBERTaV3" | "DomainBERT":
                eval_loss_fn = CrossEntropyLoss()
                if settings.enhanced_fine_tuning:
                    trained_model = DeBERTa.train_model(
                        train_dataset, eval_dataset, tokenizer
                    )
                else:
                    size = settings.bert_size
                    pretrained_model_name_or_path = f"microsoft/deberta-v3-{size}"
                    if settings.use_pretrained_bert_model:
                        pretrained_model_name_or_path = settings.pretrained_model_path
                    trained_model = huggingface_models.train_model(
                        train_dataset,
                        eval_dataset,
                        tokenizer,
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                    )

            case "DistilBERT":
                trained_model = huggingface_models.train_model(
                    train_dataset,
                    eval_dataset,
                    tokenizer,
                    pretrained_model_name_or_path="distilbert/distilbert-base-uncased",
                )

            case "RoBERTa":
                trained_model = huggingface_models.train_model(
                    train_dataset,
                    eval_dataset,
                    tokenizer,
                    pretrained_model_name_or_path="FacebookAI/roberta-base",
                )

            case "ResNet":
                trained_model = ResNet.train(train_dataset, eval_dataset, tokenizer)

        if trained_model is None:
            continue

        # Start evaluating of the finalized model
        if evaluate_idgas:
            results = idga_evaluation(
                trained_model, train_domains, tokenizer, tokenization
            )
            results["complete"] = evaluate(
                trained_model, eval_dataset, loss_fn=eval_loss_fn
            )
            results["orig"] = evaluate(
                trained_model, orig_eval_dataset, loss_fn=eval_loss_fn
            )
        elif settings.real_world_study:
            # iterate over all 46 DGAs
            logger.info("Starting Real World Evaluation")

            rw_domains, rw_labels = get_data(
                preprocess=False,
                tokenizer=tokenizer,
                pre_train=False,
                eval_different_dataset=False,
                evaluate_rw=True,
            )
            rw_domains = np.array(rw_domains)
            rw_labels = np.array(rw_labels)
            results = {}
            logger.info(f"Evaluate benign domains")
            rw_benign_domains, rw_benign_labels = zip(
                *[(x, y) for x, y in zip(rw_domains, rw_labels) if y == 0]
            )
            rw_benign_dataset = get_dataset(
                list(rw_benign_domains), list(rw_benign_labels), tokenization, tokenizer
            )

            benign_results = evaluate(
                trained_model, rw_benign_dataset, loss_fn=eval_loss_fn
            )

            results["benign"] = benign_results
            for dga in rw_eval_dgas:
                logger.info(f"Evaluation using DGA {dga}")
                # rw_dga is the real world data with only this one dga combined
                rw_dga_domains, rw_dga_labels = zip(
                    *[
                        (x, y)
                        for x, y in zip(rw_domains, rw_labels)
                        if y == label_to_int[dga]
                    ]
                )
                rw_dga_dataset = get_dataset(
                    list(rw_dga_domains), list(rw_dga_labels), tokenization, tokenizer
                )

                dga_results = evaluate(
                    trained_model, rw_dga_dataset, loss_fn=eval_loss_fn
                )
                results[dga] = dga_results
        elif settings.balanced_idn_analysis:

            logger.info("Starting IDN Evaluation")

            idn_domains, idn_labels = get_data(
                preprocess=False,
                tokenizer=tokenizer,
                pre_train=False,
                eval_different_dataset=False,
                evaluate_rw=False,
                evaluate_balanced_idn=True,
            )
            idn_domains = np.array(idn_domains)
            idn_labels = np.array(idn_labels)
            logger.info(f"Evaluate {len(idn_domains)} idn domains")
            idn_benign_dataset = get_dataset(
                list(idn_domains),
                list(idn_labels),
                tokenization,
                tokenizer,
            )
            results = {}
            results["idn_test"] = evaluate(
                trained_model, idn_benign_dataset, loss_fn=eval_loss_fn
            )
            results["complete"] = evaluate(
                trained_model, eval_dataset, loss_fn=eval_loss_fn
            )
            results["orig"] = evaluate(
                trained_model, orig_eval_dataset, loss_fn=eval_loss_fn
            )

        else:

            results = evaluate(trained_model, eval_dataset, loss_fn=eval_loss_fn)
        results["split"] = i
        evaluation_results.append(results)

    return evaluation_results


def evaluate(trained_model, eval_dataset, loss_fn=None):
    start_time = time.time()
    name = get_name()
    if loss_fn is None:
        loss_fn = nn.BCELoss()
    val_loader = DataLoader(eval_dataset, batch_size=128)
    metrics = evaluate_model(
        trained_model,
        val_loader,
        "cuda:0",
        loss_fn,
        name,
        start_time,
    )
    return metrics


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0, greater_is_better: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.greater_is_better = greater_is_better
        if self.greater_is_better:
            self.best_metric = -float("inf")
        else:
            self.best_metric = float("inf")

    def early_stop(
            self,
            metric,
    ):
        if self.greater_is_better:
            if metric > self.best_metric:
                self.best_metric = metric
                self.counter = 0
            elif metric < (self.best_metric + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if metric < self.best_metric:
                self.best_metric = metric
                self.counter = 0
            elif metric > (self.best_metric + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def train(
        name,
        hparams,
        train_dataset,
        eval_dataset,
        model_class=None,
        get_model=None,
):
    """

    :param hparams:
    :param train_dataset:
    :param eval_dataset:
    :param model_class: Set either model_class or get_model
    :param get_model: Set either model_class or get_model
    :return:
    """
    return train_model(
        name,
        hparams,
        train_dataset,
        eval_dataset,
        get_model=get_model,
        model_class=model_class,
    )


def evaluate_model(
        model, val_loader, device, loss_fn, name, start_time,
):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        loss_eval = 0
        total_labels = []
        total_orig_labels = []
        total_orig_domains = []
        all_predictions = []
        for data in val_loader:
            inputs = data["input_ids"]
            labels = data["label"]
            orig_labels = data["orig_label"]
            attention_masks = data["attention_mask"]
            total_labels.append(labels)
            total_orig_labels.append(orig_labels)
            total_orig_domains.extend(data["domain"])

            inputs, labels = torch.transpose(torch.stack(inputs, dim=0), 0, 1).to(
                device
            ), labels.to(device)
            attention_masks = torch.transpose(
                torch.stack(attention_masks, dim=0), 0, 1
            ).to(device)
            outputs = model(inputs, attention_mask=attention_masks)

            if isinstance(outputs, SequenceClassifierOutput):
                outputs = outputs.logits

            all_predictions.extend(outputs)

            if len(outputs.shape) == 1:
                loss = loss_fn(outputs, labels.float())
            else:
                num_classes = 2
                if settings.classification_task == "multiclass":
                    num_classes = len(label_to_int)
                loss = loss_fn(outputs.view(-1, num_classes), labels.view(-1))

            loss_eval += loss.item() / len(val_loader)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        total_labels = torch.cat(total_labels, dim=0)

        all_predictions = torch.stack(all_predictions)

        metrics = calculate_metrics(total_labels, all_predictions.cpu(), save=False)
        metrics["loss"] = loss_eval

        metrics_to_log = copy.deepcopy(metrics)
        metrics_to_log.pop("fpr", None)
        metrics_to_log.pop("tpr", None)
        metrics_to_log.pop("classification_report", None)
        metrics_to_log.pop("predictions")
        metrics_to_log.pop("labels")

        total_orig_labels = torch.cat(total_orig_labels, dim=0)
        metrics["orig_labels"] = total_orig_labels.tolist()
        metrics["orig_domains"] = total_orig_domains

        logger.info(metrics_to_log)

        elapsed_time = time.time() - start_time

        logger.info(f"Elapsed time: {elapsed_time} seconds")
    return metrics


def train_model(
        name,
        hparams,
        train_dataset,
        eval_dataset,
        config: dict = None,
        model_class=None,
        get_model=None,
):
    # loss function, and optimizer
    loss_fn = CrossEntropyLoss()

    if config:
        update_config(hparams, config)

    learning_rate = hparams.get("learning_rate")
    batch_size = hparams.get("batch_size")

    hparams = OmegaConf.create(hparams)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    if get_model:
        model = get_model(hparams)
    else:
        model = model_class(hparams)

    model.to(device)
    logger.info(model)

    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {num_parameters}")

    optimizer = AdamW(params=model.parameters(), lr=learning_rate, weight_decay=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(eval_dataset, batch_size=batch_size)

    num_epochs = hparams["num_epochs"]

    best_model_weights = None
    best_metrics = 0
    if settings.classification_task == "binary":
        best_metric_indicator = "roc_auc@max_fpr=0.01"
    else:
        best_metric_indicator = "f1-score"
    greater_is_better = True

    early_stopper = EarlyStopper(
        patience=hparams["early_stopping_grace"], greater_is_better=greater_is_better
    )

    # Train the model
    for epoch in range(num_epochs):
        start_time = time.time()
        logger.info(f"Epoch {epoch}")
        training_loss = 0
        start_log_time = time.time()
        for i, data in enumerate(train_loader):
            model.train()
            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            inputs = data["input_ids"]
            labels = data["label"]
            inputs, labels = torch.transpose(torch.stack(inputs, dim=0), 0, 1).to(
                device
            ), labels.to(device)
            loss = model(inputs, labels=labels)

            if isinstance(loss, SequenceClassifierOutput):
                loss = loss.loss

            # Backward pass
            loss.backward()
            training_loss += loss.item()

            # Update the parameters
            optimizer.step()
            if i != 0 and i % LOG_EVERY == 0:
                elapsed_log_time = time.time() - start_log_time
                logger.info(
                    f"training loss: {training_loss / LOG_EVERY:.5f} ({i / len(train_loader) * 100:.2f}% of Epoch {epoch}/{num_epochs}) - {elapsed_log_time:.2f} seconds since last logging"
                )
                start_log_time = time.time()
                training_loss = 0
            if EVAL_EVERY < 1:
                eval_every = math.ceil(len(train_loader) * EVAL_EVERY)
            else:
                eval_every = EVAL_EVERY
            if i != 0 and i % eval_every == 0:
                metrics = evaluate_model(
                    model, val_loader, device, loss_fn, name, start_time
                )

                if greater_is_better:
                    if best_metrics < metrics[best_metric_indicator]:
                        best_metrics = metrics[best_metric_indicator]
                        logger.info(f"Best Metric: {best_metrics}")
                        best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    if metrics[best_metric_indicator] < best_metrics:
                        best_metrics = metrics[best_metric_indicator]
                        logger.info(f"Best Metric: {best_metrics}")
                        best_model_weights = copy.deepcopy(model.state_dict())

        metrics = evaluate_model(model, val_loader, device, loss_fn, name, start_time)
        if greater_is_better:
            if best_metrics < metrics[best_metric_indicator]:
                best_metrics = metrics[best_metric_indicator]
                logger.info(f"Best Metric: {best_metrics}")
                best_model_weights = copy.deepcopy(model.state_dict())
        else:
            if metrics[best_metric_indicator] < best_metrics:
                best_metrics = metrics[best_metric_indicator]
                logger.info(f"Best Metric: {best_metrics}")
                best_model_weights = copy.deepcopy(model.state_dict())
        if early_stopper.early_stop(metrics[best_metric_indicator]):
            break

    logger.info("Finished training")
    model.load_state_dict(best_model_weights)

    logger.info("Evaluating Finished Model")
    start_time = time.time()
    evaluate_model(model, val_loader, device, loss_fn, name, start_time)
    return model


def idga_evaluation(trained_model, train_domains, tokenizer, tokenization):
    idgas = __get_all_idgas(exclude_scripts=False)
    results = {}
    logger.debug(idgas)
    for idga in idgas:
        logger.info(f"Evaluate {idga}...")
        idn_eval_data, idn_eval_labels = get_idn_data(idga=[idga], get_only_mal=True)

        filtered_data = [
            (domain, label)
            for domain, label in zip(idn_eval_data, idn_eval_labels)
            if domain not in train_domains
        ]
        if filtered_data:
            filtered_idn_eval_data, filtered_idn_eval_labels = zip(*filtered_data)

        else:
            logger.info(f"No data to evaluate?")
            results[idga] = None
            continue

        eval_idn_dataset = get_dataset(
            list(filtered_idn_eval_data),
            list(filtered_idn_eval_labels),
            tokenization,
            tokenizer,
        )

        start_time = time.time()
        name = get_name()
        loss_fn = CrossEntropyLoss()
        val_loader = DataLoader(eval_idn_dataset, batch_size=128)
        metrics = evaluate_model(
            trained_model,
            val_loader,
            "cuda:0",
            loss_fn,
            name,
            start_time,
        )
        results[idga] = metrics["acc"]

    return results



