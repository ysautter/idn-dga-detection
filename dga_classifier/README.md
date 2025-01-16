# DGA Classification Training and Evaluation

This module enables the training and evaluation of DGA classification models.

To start, execute the main.py file. The architecture, training parameters and experiments can be mostly configured using environment variables. These can also be defined in a `.env` file next to the main.py.

For a minimal working example use the following parameters.

```
FINE_TUNING=True
MODEL=ResNet
BENIGN_DATASET=/path/to/benign.csv
MAL_DATASET=/home/path/to/mal.csv
```

For an example of different pre-training, fine-tuning and IDN analysis environment checkout `.env.template`

## Environment Parameters

| Parameter                                  | Type  | Default                                                                      | Description                                                                                                                              |
|--------------------------------------------|-------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------| 
| `MODEL`                                    | str   |                                                                              | Required: Specifies the model to be trained.                                                                                             |
| `DOMAIN_TOKENIZER`                         | bool  | False                                                                        | Use the tokenizer specified using `DEBERTA_VOCAB_PATH`                                                                                   |
| `FINE_TUNING`                              | bool  | False                                                                        | If the model should fine-tuned to DGA classification                                                                                     |
| `PRE_TRAIN`                                | bool  | False                                                                        | If the model should pre-trained. Only works when `MODEL` is set to `DomainBERT` or `DeBERTaV3`                                           |
| `LOGLEVEL`                                 | str   | INFO                                                                         | Specifies the log level                                                                                                                  |
| `LOGDIR`                                   | str   | /tmp                                                                         | Specifies the log folder                                                                                                                 |
| `K_FOLDS`                                  | int   | 5                                                                            | Specifies the number k-fold cross-validation                                                                                             |
| `SINGLE_FOLD`                              | bool  | False                                                                        | Stops evaluation after the first fold                                                                                                    |
| `PRETRAINED_MODEL_PATH`                    | str   | None                                                                         | Specifies the path to the pre-trained model. The folder needs to contain the `config.json`, `pytorch_model.bin` and `tokenizer.model`                     |
| `IDGA_EVALUATION`                          | bool  | False                                                                        | Trains a model and evaluates its performance on the IDGAs                                                                                |
| `IDGA_TRAINED_EVALUATION`                  | bool  | False                                                                        | Trains a model using including the IDGAs and evaluates its performance on the IDGAs                                                      |
| `NORMAL_RUN`                               | bool  | True                                                                         | Fine-Tunes/Trains and evaluates a model on the default binary classification set                                                         |
| `BALANCED_IDN_ANALYSIS`                    | bool  | False                                                                        | Traines a model on language balances benign and malicious IDNs and evaluates its performance on the hold out IDNs and the blocklist IDNs |
| `EVALUATE_INFERENCE_TIMES`                 | bool  | False                                                                        | Evaluates inference times of BERT, DeBERTaV3-xsmall, DeBERTaV3-base, DeBERTaV3-large and ResNet                                          |
| `LEAVE_ONE_IDGA_VARIANT_OUT_ANALYSIS`      | bool  | False                                                                        | Performs the leave-one-idga-variant-out analysis                                                                                         |
| `LEAVE_ONE_IDGA_OUT_ANALYSIS`              | bool  | False                                                                        | Performs the leave-one-idga-out analysis                                                                                                 |
| `LEAVE_ONE_LANGUAGE_OUT_ANALYSIS`          | bool  | False                                                                        | Performs the leave-one-language-out analysis                                                                                             |
| `LEAVE_ONE_SCRIPT_OUT_ANALYSIS`            | bool  | False                                                                        | Performs the leave-one-script-out analysis                                                                                               |
| `INCLUDE_ONLY_ONE_IDGA_VARIANT_ANALYSIS`   | bool  | False                                                                        | Performs the include-only-one-idga-variant analysis                                                                                      |
| `INCLUDE_ONLY_ONE_IDGA_ANALYSIS`           | bool  | False                                                                        | Performs the include-only-one-idga analysis                                                                                              |
| `INCLUDE_ONLY_ONE_LANGUAGE_ANALYSIS`       | bool  | False                                                                        | Performs the include-only-one-language analysis                                                                                          |
| `INCLUDE_ONLY_ONE_script_ANALYSIS`         | bool  | False                                                                        | Performs the include-only-one-script analysis                                                                                            |
| `ENHANCED_FINE_TUNING`                     | str   | ""                                                                           | Specifies the enhanced fine-tuning method. Possible options are: `CNN`, `BiLSTM` and `ResNet`                                            |
| `ENHANCED_FINE_TUNING_FEEATURE_EXTRACTION` | bool  | False                                                                        | Whether feature extraction should be used during fine-tuning                                                                             |
| `STATISTICAL_ANALYSIS`                     | bool  | False                                                                        | Executes the normal_run 10 times using different seeds                                                                                   |
| `REAL_WORLD_STUDY`                         | bool  | False                                                                        | Executes the real world training and evaluation                                                                                          |
| `EVALUATION_FOLDER`                        | str   | "evaluations"                                                                | Under which folder the evaluation results should be stored                                                                               |
| `USE_PRETRAINED_BERT_MODEL`                | bool  | False                                                                        | Loads the model weights from `PRETRAINED_MODEL_PATH`                                                                                     |
| `TRAIN_BATCH_SIZE`                         | int   | 128                                                                          | Batch size used during training                                                                                                          |
| `EVAL_BATCH_SIZE`                          | int   | 128                                                                          | Batch size used during evaluation                                                                                                        |
| `LEARNING_RATE`                            | float | 1e-5                                                                         | Learning rate used by the Adam Optimizer during fine-tuning/training                                                                     |
| `WARMUP_RATIO`                             | float | 0.2                                                                          | Warmup ratio used during training fine-tuning                                                                                            |
| `BERT_SIZE`                                | str   | "base"                                                                       | Which DeBERTaV3 size to be used.                                                                                                         |
| `CLASSIFICATION_TASK`                      | str   | "binary"                                                                     | Which classification task the model should be trained on. Options: `binary` or `multiclass`                                              |
| `CHECKPOINTS_FOLDER`                       | str   | "checkpoints"                                                                | Under which folder checkpoints should be stored                                                                                          |
| `PROJECT_NAME`                             | str   | "DGA Classifier"                                                             | Is used to name evaluation files, checkpoints and wandb projects                                                                         |
| `DEBERTA_DATA_DIR`                         | str   | None                                                                         | Which data should be used in the pre-training of DeBERTaV3. The folder must contain a `train.txt` and a `valid.txt`                      |
| `DEBERTA_TRAIN_BATCH_SIZE`                 | int   | 256                                                                          | Batch size used in the pre-training of DeBERTaV3                                                                                         |
| `DEBERTA_EVAL_BATCH_SIZE`                  | int   | 256                                                                          | Batch size used in the evaluation process of the pre-training of DeBERTaV3                                                               |
| `DEBERTA_CONFIG`                           | str   | "models/DeBERTa/DeBERTa_official/experiments/language_model/rtd_xsmall.json" | Specifies the config file used by DeBERTaV3. Defaults to DeBERTaV3-xsmall configuration.                                                 |
| `DEBERTA_PRETRAINED_DISCRIMINATOR_PATH`    | str   | ""                                                                           | Path to the pre-trained DeBERTaV3 discriminator, which should be used for continued pre-training                                         |
| `DEBERTA_PRETRAINED_GENERATOR_PATH`        | str   | ""                                                                           | Path to the pre-trained DeBERTaV3 generator, which should be used for continued pre-training                                             |
| `DEBERTA_VOCAB_PATH`                       | str   | None                                                                         | Path to the tokenizer used during pre-training                                                                                           |
| `DEBERTA_NUM_TRAINING_STEPS`               | int   | 1_000_000                                                                    | Number of training steps to be performed during pre-training                                                                             |

### Available Models

New models can be added to `dga_classifier/models`. Most of the BERT models are loaded via the `huggingface_models.py` module. The structure of this folder has been mostly used during the earlier stages of development. Since most of them relied on other repos which were added at the beginning via the git submodules, most of them do not work anymore. That's why they are commented out in the `main.py`.

In the following, the possible `MODEL` parameter values are listed:

| Model       | Description                                                                                                                                                                                                                                    |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| ResNet      | Uses the E2LD-ResNet architecture when `CLASSIFICATION_TASK` is set to `binary` and the M-ResNet architecture when it is set to `multiclass`.                                                                                                  | 
| DomainBERT  | Is an alias for DeBERTaV3, but should be used in combination with `DEBERTA_VOCAB_PATH`, `PRETRAINED_MODEL_PATH`, `DOMAIN_TOKENIZER` and `USE_PRETRAINED_BERT_MODEL`.                                                                           | 
| Encoder     | Utilizes the [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture but does not initialize the training. It utilizes the [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) tokenizer. |
| BERT        | Loads the [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) weights and loads the corresponding tokenizer.                                                                                                             |
| DistilBERT  | Loads the [DistilBERT-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) weights and loads the corresponding tokenizer.                                                                                                  |
| RoBERTa     | Loads the [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) weights and loads the corresponding tokenizer.                                                                                                                        |
| XLM-RoBERTa | Loads the [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base) weights and loads the corresponding tokenizer.                                                                                                                |
| DeBERTaV3   | Loads the [DeBERTaV3-base](https://huggingface.co/microsoft/deberta-v3-base) weights and loads the corresponding tokenizer.                                                                                                                    |

## Pre-Training Data
The pre-training data is set using the `DEBERTA_DATA_DIR` environment variable. The folder defined with this variable needs to contain the files `train.txt` and `valid.txt`. Both files are produced using the tokenizer defined using the `DEBERTA_VOCAB_PATH` variable.

## IDN Data


