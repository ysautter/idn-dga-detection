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

import torch

# Check if _six module exists, if not, create and patch it
if not hasattr(torch, '_six'):
    class _DummySix:
        string_classes = str


    torch._six = _DummySix()

# If _six is there but lacks string_classes, add it
if not hasattr(torch._six, 'string_classes'):
    torch._six.string_classes = str

import logging
import pathlib

import numpy as np

from transformers import (
    AutoTokenizer,
    BertTokenizer,
)

from dga_classifier import get_settings, save_metrics, get_name, set_seed
from dga_classifier.config import LogConfig
from logging.config import dictConfig


from dga_classifier.data.dataset import (
    char_to_int,
)
from dga_classifier.experiments import run_idga_trained_evaluation, run_idga_evaluation, \
    run_include_only_one_script_analysis, run_leave_one_script_out_analysis, run_include_only_one_language_analysis, \
    run_leave_one_language_out_analysis, run_include_only_one_idga_analysis, run_leave_one_idga_out_analysis, \
    run_leave_one_idga_variant_out_analysis, run_include_only_one_idga_variant_analysis, run_real_world_evaluation, \
    run_statistical_analysis, run_balanced_idn_analysis
from dga_classifier.models import DeBERTa

from dga_classifier.trainer import run

settings = get_settings()
log_config = LogConfig(LOG_LEVEL=settings.loglevel, LOGDIR=settings.logdir)
dictConfig(log_config.model_dump())

logger = logging.getLogger(log_config.LOGGER_NAME)

logger.info(f"Starting Master Thesis Project")
logger.info(f"Logging to {settings.logdir} with log level {settings.loglevel}")


pathlib.Path("evaluations/idga").mkdir(parents=True, exist_ok=True)

set_seed(42)


def main():
    # Setting Up Models
    model = settings.model
    logger.info(f"Selected model {model}")

    if settings.pre_training:
        return DeBERTa.pre_train()

    tokenizer = None
    if settings.domain_tokenizer:
        tokenizer = DeBERTa.get_tokenizer()
    else:
        if "RoBERTa" == model and settings.fine_tuning:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        if "BERT" == model or "Encoder" == model and settings.fine_tuning:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if "DeBERTaV3" == model or "DomainBERT" == model:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        if "DistilBERT" == model and settings.fine_tuning:
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        if "XLM-Roberta" == model and settings.fine_tuning:
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    if settings.classification_task == "binary":
        model_max_length = 63  # 253 + start and end token?
    else:
        model_max_length = 253

    def tokenization(example):
        example = example["domain"]
        example = [el.lower() for el in example]
        if tokenizer:
            return tokenizer(example, padding="max_length", max_length=model_max_length)
        else:
            element = example
            mapping = list(map(char_to_int.get, element))
            length = model_max_length - len(mapping)
            padded = np.pad(mapping, (0, length)).tolist()
            processed_element = {
                "input_ids": padded,
                "attention_mask": [el if el == 0 else 1 for el in padded],
                "token_type_ids": [0] * len(padded),
            }
            return processed_element

    if settings.include_only_one_idga_variant_analysis:
        run_include_only_one_idga_variant_analysis(model, tokenizer, tokenization)

    if settings.leave_one_idga_variant_out_analysis:
        run_leave_one_idga_variant_out_analysis(model, tokenizer, tokenization)

    if settings.leave_one_idga_out_analysis:
        run_leave_one_idga_out_analysis(model, tokenizer, tokenization)

    if settings.include_only_one_idga_analysis:
        run_include_only_one_idga_analysis(model, tokenizer, tokenization)

    if settings.leave_one_language_out_analysis:
        run_leave_one_language_out_analysis(model, tokenizer, tokenization)

    if settings.include_only_one_language_analysis:
        run_include_only_one_language_analysis(model, tokenizer, tokenization)

    if settings.leave_one_script_out_analysis:
        run_leave_one_script_out_analysis(model, tokenizer, tokenization)

    if settings.include_only_one_script_analysis:
        run_include_only_one_script_analysis(model, tokenizer, tokenization)

    if settings.idga_evaluation:
        run_idga_evaluation(model, tokenizer, tokenization)

    if settings.idga_trained_evaluation:
        run_idga_trained_evaluation(model, tokenizer, tokenization)

    if settings.normal_run:
        result = run(model, tokenizer, tokenization, evaluate_idgas=False)
        name = get_name()
        save_metrics(name, result)

    if settings.real_world_study:
        run_real_world_evaluation(model, tokenizer, tokenization)

    if settings.statistical_analysis:
        run_statistical_analysis(model, tokenizer, tokenization)

    if settings.balanced_idn_analysis:
        run_balanced_idn_analysis(model, tokenizer, tokenization)


if __name__ == "__main__":
    main()
