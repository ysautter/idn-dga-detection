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

import os
from typing import List, Optional, Self

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    model: str
    @field_validator("model", mode="after")
    @classmethod
    def valid_model(cls, value: str):
        allowed_models = ["ResNet", "RoBERTa", "BERT", "DeBERTaV3", "DistilBERT", "XLM-RoBERTa", "DomainBERT"]
        if value not in allowed_models:
            raise ValueError(f"{value} is not an allowed model. Allowed models are {allowed_models}.")
        return value

    bert_size: str = "base"

    classification_task: str = "binary"
    @field_validator("classification_task", mode="after")
    @classmethod
    def valid_classification_task(cls, value: str):
        allowed_classification_task = ["multiclass", "binary"]
        if value not in allowed_classification_task:
            raise ValueError(f"{value} is not an allowed classification_task. "
                             f"Allowed classification tasks are {allowed_classification_task}.")
        return value

    # Datasets
    benign_dataset: Optional[str] = None
    mal_dataset: Optional[str] = None

    benign_multiclass_dataset: Optional[str] = None
    mal_multiclass_dataset: Optional[str] = None

    benign_rw_dataset: Optional[str] = None
    mal_rw_dataset: Optional[str] = None
    benign_rw_test_dataset: Optional[str] = None
    mal_rw_test_dataset: Optional[str] = None

    # IDN Datasets
    idn_data_mal: Optional[str] = None
    idn_data_benign: Optional[str] = None

    # Hold out IDN test split for balanced language evaluation
    idn_data_benign_test: Optional[str] = None
    idn_data_mal_test: Optional[str] = None

    # Logging
    loglevel: str = "INFO"
    logdir: str = "./"

    # Training and evaluation settings
    domain_tokenizer: bool = False
    fine_tuning: bool = False
    pre_training: bool = False
    k_folds: int = 5
    single_fold: bool = False

    pretrained_model_path: Optional[str] = None
    use_pretrained_bert_model: bool = False

    idga_evaluation: bool = False  # Use for generating ∅
    idga_trained_evaluation: bool = False  # Use for generating U
    normal_run: bool = True

    balanced_idn_analysis: bool = False

    leave_one_idga_variant_out_analysis: bool = False
    leave_one_idga_out_analysis: bool = False
    leave_one_language_out_analysis: bool = False
    leave_one_script_out_analysis: bool = False

    loivo_index: int = 0
    iooiv_index: int = 0
    loivo_split_n: int = 100

    include_only_one_language_analysis: bool = False
    include_only_one_idga_analysis: bool = False
    include_only_one_idga_variant_analysis: bool = False
    include_only_one_script_analysis: bool = False

    enhanced_fine_tuning: Optional[str] = None
    @field_validator("enhanced_fine_tuning", mode="after")
    @classmethod
    def valid_enhanced_fine_tuning(cls, value: str):
        if value is None:
            return value
        if value not in ["CNN", "BiLSTM", "ResNet"]:
            raise ValueError(f"{value} is not a valid enhanced fine_tuning mode")
        return value

    enhanced_fine_tuning_feature_extraction: bool = False

    statistical_analysis: bool = False

    real_world_study: bool = False

    resume_from_checkpoint: bool = False

    train_batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.2

    deberta_data_dir: Optional[str] = None
    deberta_train_batch_size: int = 256
    deberta_eval_batch_size: int = 256
    deberta_config: Optional[str] = (
        "models/DeBERTa/DeBERTa_official/experiments/language_model/rtd_xsmall.json"
    )

    deberta_pretrained_discriminator_path: Optional[str] = ""
    deberta_pretrained_generator_path: Optional[str] = ""

    deberta_vocab_path: Optional[str] = None
    deberta_num_training_steps: int = 1_000_000


    project_name: str = "DGA Classifier"
    checkpoints_folder: str = "checkpoints"
    evaluation_folder: str = "evaluations"

    def __check_idn_training(self):
        return any([
            self.idga_evaluation,
            self.idga_trained_evaluation,
            self.balanced_idn_analysis,
            self.leave_one_idga_variant_out_analysis,
            self.leave_one_idga_out_analysis,
            self.leave_one_language_out_analysis,
            self.leave_one_script_out_analysis,
            self.include_only_one_language_analysis,
            self.include_only_one_idga_analysis,
            self.include_only_one_idga_variant_analysis,
            self.include_only_one_script_analysis,
        ])

    @model_validator(mode="after")
    def check_mismatch(self) -> Self:
        if self.use_pretrained_bert_model and self.pretrained_model_path is None:
            raise ValueError("If USE_PRETRAINED_BERT_MODEL is True, "
                             "the pretrained model is required by setting PRETRAINED_MODEL_PATH.")

        if self.domain_tokenizer and self.deberta_vocab_path is None:
            raise ValueError("If DOMAIN_TOKENIZER is True, "
                             "the vocabulary is required by setting DEBERTA_VOCAB_PATH.")

        if self.real_world_study and (self.benign_rw_dataset is None or
                                      self.mal_rw_dataset is None or
                                      self.benign_rw_test_dataset is None or
                                      self.mal_rw_test_dataset is None):
            raise ValueError("When performing the real-world study the datasets BENIGN_RW_DATASET, "
                             "MAL_RW_DATASET, BENIGN_RW_TEST_DATASET and MAL_RW_TEST_DATASET are required.")

        if self.real_world_study and self.mal_rw_dataset is not None and not self.mal_rw_dataset.endswith("_x.csv"):
            raise ValueError("When performing the real-world study the dataset MAL_RW_DATASET "
                             "needs to end with _x.csv, as x is substituted by numbers from 0 to 9")

        if (self.fine_tuning and self.__check_idn_training() and
                (self.idn_data_benign is None or self.idn_data_mal is None)):
            raise ValueError("When performing FINE_TUNING in IDN experiments, "
                             "IDN_DATA_BENIGN and IDN_DATA_MAL are required.")

        if self.fine_tuning and self.pre_training:
            raise ValueError("You can either perform FINE_TUNING or PRE_TRAINING but not both at the same time.")

        if self.balanced_idn_analysis and (self.idn_data_benign_test is None or self.idn_data_mal_test is None):
            raise ValueError("When performing the BALANCED_IDN_ANALYSIS the datasets IDN_DATA_BENIGN, IDN_DATA_MAL, "
                             "IDN_DATA_BENIGN_TEST and IDN_DATA_MAL_TEST are required.")

        if self.fine_tuning and self.classification_task == "binary" and not self.real_world_study and (self.benign_dataset is None or self.mal_dataset is None):
            raise ValueError("When performing FINE_TUNING on the binary classification task, "
                             "BENIGN_DATASET and MAL_DATASET are required.")

        if self.fine_tuning and self.classification_task == "multiclass" and (self.benign_multiclass_dataset is None or self.mal_multiclass_dataset is None):
            raise ValueError("When performing FINE_TUNING on the multiclass classification task, "
                             "BENIGN_MULTICLASS_DATASET and MAL_MULTICLASS_DATASET are required.")

        if self.pre_training and (self.model != "DomainBERT" and self.model != "DeBERTaV3"):
            raise ValueError("Pretraining is only available for DomainBERT and DeBERTaV3.")

        if self.pre_training and (self.deberta_vocab_path is None or self.deberta_data_dir is None):
            raise ValueError("When performing PRE_TRAINING, the vocabulary "
                             "DEBERTA_VOCAB_PATH and training data DEBERTA_DATA_DIR is required.")

        if self.enhanced_fine_tuning is not None and self.model != "DeBERTaV3":
            raise ValueError("Enhanced fine-tuning is only available for DeBERTaV3.")

        if self.enhanced_fine_tuning_feature_extraction and self.model != "DeBERTaV3":
            raise ValueError("Enhanced fine-tuning feature extraction is only available for DeBERTaV3.")

        if self.model == "DomainBERT" and self.fine_tuning:
            if self.deberta_vocab_path is None:
                raise ValueError("When fine-tuning DomainBERT also provide the tokenizer using DEBERTA_VOCAB_PATH.")
            if self.pretrained_model_path is None:
                raise ValueError("When fine-tuning DomainBERT provide the path to the pre-trained model"
                                 " using PRETRAINED_MODEL_PATH.")
            if not self.domain_tokenizer:
                raise ValueError("When fine-tuning DomainBERT you must also set DOMAIN_TOKENIZER to True.")

            if not self.use_pretrained_bert_model:
                raise ValueError("When fine-tuning DomainBERT you must also set USE_PRETRAINED_BERT_MODEL to True.")

        return self



class LogConfig(BaseModel):
    """Logging configuration to be set for dga classifier"""

    LOGGER_NAME: str = "dga_classifier"
    LOG_FORMAT: str = "%(levelname)s | %(name)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "INFO"
    LOGDIR: str = "./"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "logging.Formatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "stream": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "formatter": "default",
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGDIR, "dga_classifier.log"),
        },
    }
    root: dict = {
        "level": LOG_LEVEL,
        "handlers": ["stream", "file"],
    }
