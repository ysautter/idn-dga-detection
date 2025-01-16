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
import os.path
import time
import uuid

import torch

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    BertConfig,
    BertForSequenceClassification,
)

from dga_classifier import calculate_metrics, get_settings, get_name
from dga_classifier.data.dataset import label_to_int

logger = logging.getLogger("dga_classifier.huggingface_models")

settings = get_settings()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # prediction_probs = torch.nn.functional.softmax(torch.Tensor(logits), dim=-1)[:, 1]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    metrics = calculate_metrics(labels, logits, save=False)
    metrics_to_log = copy.deepcopy(metrics)
    metrics_to_log.pop("fpr", None)
    metrics_to_log.pop("tpr", None)
    metrics_to_log.pop("classification_report", None)
    metrics_to_log.pop("predictions")
    metrics_to_log.pop("labels")
    metrics_to_log.pop("name")

    logger.info(metrics_to_log)
    return metrics_to_log


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def train_model(
    train_dataset,
    eval_dataset,
    tokenizer,
    pretrained_model_name_or_path: str = "google-bert/bert-base-uncased",
):
    if pretrained_model_name_or_path == "":
        logger.info(f"Training model without pretraining")
    else:
        logger.info(f"Loading model from {pretrained_model_name_or_path}")
    metric_for_best_model = "eval_roc_auc@max_fpr=0.01"
    if settings.classification_task == "multiclass":
        metric_for_best_model = "f1-score"

    if pretrained_model_name_or_path[0] == "/":
        output_dir = (
            f"{settings.checkpoints_folder}/finetuning{pretrained_model_name_or_path}"
        )
    else:
        output_dir = (
            f"{settings.checkpoints_folder}/finetuning/{pretrained_model_name_or_path}"
        )
    output_dir = os.path.join(output_dir, get_name(), str(uuid.uuid4()))

    logger.info(f"Output folder: {output_dir}")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        learning_rate=settings.learning_rate,
        weight_decay=0.01,
        num_train_epochs=20,
        logging_steps=100,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # save_strategy="steps",
        # save_steps=10000,
        # evaluation_strategy="steps",
        # eval_steps=50,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model=metric_for_best_model,
        per_device_eval_batch_size=settings.eval_batch_size,
        per_device_train_batch_size=settings.train_batch_size,
        disable_tqdm=True,
        lr_scheduler_type="linear",
        warmup_ratio=settings.warmup_ratio,
    )
    num_labels = 2
    if settings.classification_task == "multiclass":
        num_labels = len(label_to_int)

    if pretrained_model_name_or_path != "":
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            # problem_type="single_label_classification",
        )
    else:
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        model = BertForSequenceClassification(config)
    model.to("cuda:0")
    logger.info(model)

    parameters_amount = sum(p.numel() for p in model.parameters())

    logger.info(f"Number of parameters {parameters_amount}")

    # Freeze layers
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    train_dataset = train_dataset.remove_columns("domain")
    train_dataset = train_dataset.remove_columns("orig_label")
    eval_dataset = eval_dataset.remove_columns("domain")
    eval_dataset = eval_dataset.remove_columns("orig_label")
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logger.info("Start training the model...")
    trainer.train()

    logger.info("Start evaluating the model..")
    trainer.evaluate()

    logger.info("Saving the model...")
    trainer.save_model()

    # Because run many in parallel saving the model for each run creates a lot of data,
    # because we don't care about the checkpoints, we delete it for this experiment
    if settings.leave_one_idga_variant_out_analysis:
        try:
            os.rmdir(output_dir)
            print(f"Directory {output_dir} deleted.")
        except OSError as e:
            print(f"Error: {e.strerror}")

    return trainer.model

    # for param in model.base_model.parameters():
    #     param.requires_grad = True
    #
    # training_arguments = TrainingArguments(
    #     output_dir="checkpoints/roberta",
    #     do_train=True,
    #     do_eval=True,
    #     remove_unused_columns=False,
    #     learning_rate=5e-5,
    #     weight_decay=0.01,
    #     num_train_epochs=5,
    #     logging_steps=10,
    #     save_strategy="steps",
    #     save_steps=500,
    #     evaluation_strategy="steps",
    #     eval_steps=500,
    #     load_best_model_at_end=True,
    #     greater_is_better=False,
    #     metric_for_best_model="eval_loss",
    #     per_device_eval_batch_size=16,
    #     per_device_train_batch_size=16,
    #     disable_tqdm=True,
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_arguments,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    # )
    #
    # trainer.train()
    #
    # trainer.evaluate()
    #
    # trainer.save_model()
