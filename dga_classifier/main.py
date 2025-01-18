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
    logger.info(f"Selected model {settings.model}")

    if settings.pre_training:
        return DeBERTa.pre_train()

    if settings.include_only_one_idga_variant_analysis:
        run_include_only_one_idga_variant_analysis()

    if settings.leave_one_idga_variant_out_analysis:
        run_leave_one_idga_variant_out_analysis()

    if settings.leave_one_idga_out_analysis:
        run_leave_one_idga_out_analysis()

    if settings.include_only_one_idga_analysis:
        run_include_only_one_idga_analysis()

    if settings.leave_one_language_out_analysis:
        run_leave_one_language_out_analysis()

    if settings.include_only_one_language_analysis:
        run_include_only_one_language_analysis()

    if settings.leave_one_script_out_analysis:
        run_leave_one_script_out_analysis()

    if settings.include_only_one_script_analysis:
        run_include_only_one_script_analysis()

    if settings.idga_evaluation:
        run_idga_evaluation()

    if settings.idga_trained_evaluation:
        run_idga_trained_evaluation()

    if settings.normal_run:
        result = run()
        name = get_name()
        save_metrics(name, result)

    if settings.real_world_study:
        run_real_world_evaluation()

    if settings.statistical_analysis:
        run_statistical_analysis()

    if settings.balanced_idn_analysis:
        run_balanced_idn_analysis()


if __name__ == "__main__":
    main()
