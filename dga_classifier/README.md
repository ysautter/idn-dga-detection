# DGA Classification Training and Evaluation

This module contains the training and evaluation procedures of the DGA classification models.

To start, execute the main.py file. The architecture, training parameters and experiments can be mostly configured using environment variables. These can also be defined in a `.env` file next to the main.py.

For a minimal working example use the following parameters.

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../"
export FINE_TUNING=True
export MODEL=DeBERTaV3
export BENIGN_DATASET=data/mock/fine-tuning/binary/benign_mock_data.csv
export MAL_DATASET=data/mock/fine-tuning/binary/mal_mock_data.csv
python main.py
```

For an example of different pre-training, fine-tuning and IDN analysis environments checkout `.env.template`

## Environment Parameters

| Parameter                                  | Type  | Default                                                                      | Description                                                                                                                            |
|--------------------------------------------|-------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------| 
| `MODEL`                                    | str   |                                                                              | Required: Specifies the model to be trained and evaluated.                                                                             |
| `FINE_TUNING`                              | bool  | False                                                                        | If the model should fine-tuned to DGA classification                                                                                   |
| `PRE_TRAINING`                             | bool  | False                                                                        | If the model should pre-trained. Only works when `MODEL` is set to `DomainBERT` or `DeBERTaV3`                                         |
| `LOGLEVEL`                                 | str   | INFO                                                                         | Specifies the log level                                                                                                                |
| `LOGDIR`                                   | str   | /tmp                                                                         | Specifies the log folder                                                                                                               |
| `K_FOLDS`                                  | int   | 5                                                                            | Specifies the number k-fold cross-validation                                                                                           |
| `SINGLE_FOLD`                              | bool  | False                                                                        | Stops evaluation after the first fold                                                                                                  |
| `PRETRAINED_MODEL_PATH`                    | str   | None                                                                         | Specifies the path to the pre-trained model. The folder needs to contain the `config.json`, `pytorch_model.bin` and `tokenizer.model`  |
| `IDGA_EVALUATION`                          | bool  | False                                                                        | Trains a model and evaluates its performance on the IDGAs                                                                              |
| `IDGA_TRAINED_EVALUATION`                  | bool  | False                                                                        | Trains a model using including the IDGAs and evaluates its performance on the IDGAs                                                    |
| `NORMAL_RUN`                               | bool  | True                                                                         | Fine-Tunes/Trains and evaluates a model on the default binary classification set                                                       |
| `BALANCED_IDN_ANALYSIS`                    | bool  | False                                                                        | Traines a model on language balances benign and malicious IDNs and evaluates its performance on the hold out IDNs and the blocklist IDNs |
| `EVALUATE_INFERENCE_TIMES`                 | bool  | False                                                                        | Evaluates inference times of BERT, DeBERTaV3-xsmall, DeBERTaV3-base, DeBERTaV3-large and ResNet                                        |
| `LEAVE_ONE_IDGA_VARIANT_OUT_ANALYSIS`      | bool  | False                                                                        | Performs the leave-one-idga-variant-out analysis                                                                                       |
| `LEAVE_ONE_IDGA_OUT_ANALYSIS`              | bool  | False                                                                        | Performs the leave-one-idga-out analysis                                                                                               |
| `LEAVE_ONE_LANGUAGE_OUT_ANALYSIS`          | bool  | False                                                                        | Performs the leave-one-language-out analysis                                                                                           |
| `LEAVE_ONE_SCRIPT_OUT_ANALYSIS`            | bool  | False                                                                        | Performs the leave-one-script-out analysis                                                                                             |
| `INCLUDE_ONLY_ONE_IDGA_VARIANT_ANALYSIS`   | bool  | False                                                                        | Performs the include-only-one-idga-variant analysis                                                                                    |
| `INCLUDE_ONLY_ONE_IDGA_ANALYSIS`           | bool  | False                                                                        | Performs the include-only-one-idga analysis                                                                                            |
| `INCLUDE_ONLY_ONE_LANGUAGE_ANALYSIS`       | bool  | False                                                                        | Performs the include-only-one-language analysis                                                                                        |
| `INCLUDE_ONLY_ONE_script_ANALYSIS`         | bool  | False                                                                        | Performs the include-only-one-script analysis                                                                                          |
| `ENHANCED_FINE_TUNING`                     | str   | ""                                                                           | Specifies the enhanced fine-tuning method. Possible options are: `CNN`, `BiLSTM` and `ResNet`                                          |
| `ENHANCED_FINE_TUNING_FEEATURE_EXTRACTION` | bool  | False                                                                        | Whether feature extraction should be used during fine-tuning                                                                           |
| `STATISTICAL_ANALYSIS`                     | bool  | False                                                                        | Executes the normal_run 10 times using different seeds                                                                                 |
| `REAL_WORLD_STUDY`                         | bool  | False                                                                        | Executes the real world training and evaluation                                                                                        |
| `EVALUATION_FOLDER`                        | str   | "evaluations"                                                                | Under which folder the evaluation results should be stored                                                                             |
| `USE_PRETRAINED_BERT_MODEL`                | bool  | False                                                                        | Loads the model weights from `PRETRAINED_MODEL_PATH`                                                                                   |
| `TRAIN_BATCH_SIZE`                         | int   | 128                                                                          | Batch size used during training                                                                                                        |
| `EVAL_BATCH_SIZE`                          | int   | 128                                                                          | Batch size used during evaluation                                                                                                      |
| `LEARNING_RATE`                            | float | 1e-5                                                                         | Learning rate used by the Adam Optimizer during fine-tuning/training                                                                   |
| `WARMUP_RATIO`                             | float | 0.2                                                                          | Warmup ratio used during training fine-tuning                                                                                          |
| `BERT_SIZE`                                | str   | "base"                                                                       | Which DeBERTaV3 size to be used.                                                                                                       |
| `CLASSIFICATION_TASK`                      | str   | "binary"                                                                     | Which classification task the model should be trained on. Options: `binary` or `multiclass`                                            |
| `CHECKPOINTS_FOLDER`                       | str   | "checkpoints"                                                                | Under which folder checkpoints should be stored                                                                                        |
| `PROJECT_NAME`                             | str   | "DGA Classifier"                                                             | Is used to name evaluation files, checkpoints and wandb projects                                                                       |
| `DEBERTA_DATA_DIR`                         | str   | None                                                                         | Which data should be used in the pre-training of DeBERTaV3. The folder must contain a `train.txt` and a `valid.txt`                    |
| `DEBERTA_TRAIN_BATCH_SIZE`                 | int   | 256                                                                          | Batch size used in the pre-training of DeBERTaV3                                                                                       |
| `DEBERTA_EVAL_BATCH_SIZE`                  | int   | 256                                                                          | Batch size used in the evaluation process of the pre-training of DeBERTaV3                                                             |
| `DEBERTA_CONFIG`                           | str   | "models/DeBERTa/DeBERTa_official/experiments/language_model/rtd_xsmall.json" | Specifies the config file used by DeBERTaV3. Defaults to DeBERTaV3-xsmall configuration.                                               |
| `DEBERTA_PRETRAINED_DISCRIMINATOR_PATH`    | str   | ""                                                                           | Path to the pre-trained DeBERTaV3 discriminator, which should be used for continued pre-training                                       |
| `DEBERTA_PRETRAINED_GENERATOR_PATH`        | str   | ""                                                                           | Path to the pre-trained DeBERTaV3 generator, which should be used for continued pre-training                                           |
| `DEBERTA_VOCAB_PATH`                       | str   | None                                                                         | Path to the tokenizer used during pre-training                                                                                         |
| `DOMAIN_TOKENIZER`                         | bool  | False                                                                        | Use the tokenizer specified using `DEBERTA_VOCAB_PATH`                                                                                 |
| `DEBERTA_NUM_TRAINING_STEPS`               | int   | 1_000_000                                                                    | Number of training steps to be performed during pre-training                                                                           |



### Available Models

New models can be added to `dga_classifier/models`. Most of the BERT models are loaded via the `huggingface_models.py` module. 

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
## Datasets

Additionally, the following environment variables are set, to configure the datasets:

### Pre-Training Data
The pre-training data is set using the `DEBERTA_DATA_DIR` environment variable.
The folder defined with this variable needs to contain the files `train.txt` and `valid.txt`.
Both files should have been produced using the tokenizer defined using the `DEBERTA_VOCAB_PATH` variable.
See `dga_classifier/models/DeBERTa/DeBERTa_official/experiments/language_model/prepare_data.py` and the corresponding [README](dga_classifier/models/DeBERTa/DeBERTa_official/experiments/language_model/README.md) for an example on how to tokenize the data for pre-training.

Mock data can be found in `data/mock/pre-training`.

### Fine-Tuning Data
The Fine-Tuning data should be provided as `csv` file. The domains column must not be tokenized. Note, the label for benign data should be `benign` and for the malicious data have a look at the DGA labels defined in `data.dataset.label_to_int`.

#### Binary Classification
There should only be two labels for the binary classification task.

| Dataset | Description                   |
| ------- |-------------------------------|
| BENIGN_DATASET | Two columns: Domains, Label. The label is in this case always `benign` |
| MAL_DATASET | Two columns: Domains, Label   |

Mock data can be found in `data/mock/fine-tuning`.

#### Multiclass Classification
For the multiclass classification task the label determines the originating DGA of the domain or if the domain is considered benign.

| Dataset                   | Description                   |
|---------------------------|-------------------------------|
| BENIGN_MULTICLASS_DATASET | Two columns: Domains, Label. The label is in this case always `benign` |
| MAL_MULTICLASS_DATASET    | Two columns: Domains, Label   |

Mock data can be found in `data/mock/fine-tuning/multiclass`.

#### Real-World Data
The Real-World data should be given as `csv` file. The domains column must not be tokenized.
The data provided in BENIGN_RW_DATASET and MAL_RW_DATASET should be served as independent, unbiased sources and should have no temporal or spatial overlap with the data in BENIGN_RW_TEST_DATASET and MAL_RW_TEST_DATASET.


| Dataset                | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| BENIGN_RW_DATASET      | Two columns: Domains, Label. The label is in this case always `benign` |
| MAL_RW_DATASET         | Two columns: Domains, Label                                            |
| BENIGN_RW_TEST_DATASET | Two columns: Domains, Label.  The label is in this case always `benign`                                          |
| MAL_RW_TEST_DATASET    | Two columns: Domains, Label                                            |

Mock data can be found in `data/mock/fine-tuning/binary/rw`.

#### IDN Data
The IDN data should be given as `csv` file.

| Dataset | Description                                                                        |
| ------- |------------------------------------------------------------------------------------| 
| IDN_DATA_BENIGN | Three columns: Domains, Label, Language. The label is in this case always `benign` |
| IDN_DATA_MAL | Two columns: Domains, Label. This can be produced by `idga/generate_iagd.py`       |
| IDN_DATA_BENIGN_TEST | Three columns: Domains, Label, Language. The label is in this case always `benign` |
| IDN_DATA_MAL_TEST | Two columns: Domains, Label. The label in this case should be `blocklist`           |

Mock data can be found in `data/mock/fine-tuning/binary/idn`.



