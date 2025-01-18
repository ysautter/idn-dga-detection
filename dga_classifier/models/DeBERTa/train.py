import copy
import logging
import os
import shutil

import evaluate
from pydantic import BaseModel


from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from dga_classifier import get_settings, get_name, calculate_metrics

import torch

from dga_classifier.models.DeBERTa.DeBERTa_official.DeBERTa.apps.run import (
    run_eval,
    create_model,
)
# Although load_tasks and rtd_task are not used here, these modules need to initialize which is done during import
from dga_classifier.models.DeBERTa.DeBERTa_official.DeBERTa.apps.tasks import (
    load_tasks,
    rtd_task,
    get_task,
)
from dga_classifier.models.DeBERTa.DeBERTa_official.DeBERTa.deberta import load_vocab, tokenizers
from dga_classifier.models.DeBERTa.DeBERTa_official.DeBERTa.training import (
    initialize_distributed,
)
from dga_classifier.models.DeBERTa.DeBERTa_official.DeBERTa.apps.run import (
    train_model as pretrain_deberta_model,
)
from dga_classifier.models.DeBERTa.model import (
    DeBERTaForDGAClassification,
    DeBERTaCNNForDGAClassification,
    DeBERTaCNN12ForDGAClassification,
    DeBERTaBiLSTMForDGAClassification,
    DeBERTaResNetForDGAClassification,
)

logger = logging.getLogger("dga_classifier")

accuracy = evaluate.load("accuracy")

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

    # pred_ids = torch.argmax(logits, dim=-1)
    return logits, labels


metric_for_best_model = "eval_roc_auc@max_fpr=0.01"
if settings.classification_task == "multiclass":
    metric_for_best_model = "f1-score"
training_arguments = TrainingArguments(
    output_dir=f"{settings.checkpoints_folder}/finetuning/deberta/{get_name()}",
    do_train=True,
    do_eval=True,
    remove_unused_columns=False,
    # learning_rate=8e-6,
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


def get_model(config):
    if settings.enhanced_fine_tuning == "CNN":
        model = DeBERTaCNNForDGAClassification()
        return model
    if settings.enhanced_fine_tuning == "CNN12":
        model = DeBERTaCNN12ForDGAClassification()
        return model
    if settings.enhanced_fine_tuning == "BiLSTM":
        model = DeBERTaBiLSTMForDGAClassification()
        return model
    if settings.enhanced_fine_tuning == "ResNet":
        model = DeBERTaResNetForDGAClassification()
        return model

    model = DeBERTaForDGAClassification()
    return model


def train_model(train_dataset, eval_dataset, tokenizer):
    model = get_model(None)
    logger.info(model)

    # Freeze layers
    if settings.enhanced_fine_tuning_feature_extraction:
        logger.info("Freezing BERT parameters...")
        for name, param in model.named_parameters():
            if name.startswith("deberta"):  # classifier layer
                param.requires_grad = False

        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.requires_grad}")

        training_arguments.learning_rate = 2e-4
        training_arguments.output_dir = (
            f"{settings.checkpoints_folder}/finetuning/deberta/{get_name()}-fe-1"
        )


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
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        data_collator=(DataCollatorWithPadding(tokenizer=tokenizer)),
    )

    trainer.train()

    trainer.evaluate()

    trainer.save_model()

    # Unfreeze layers
    if settings.enhanced_fine_tuning_feature_extraction:
        logger.info("Unfreezing BERT parameters...")
        for name, param in model.named_parameters():
            if name.startswith("deberta"):  # classifier layer
                param.requires_grad = True
        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.requires_grad}")

        training_arguments.learning_rate = settings.learning_rate
        training_arguments.output_dir = (
            f"{settings.checkpoints_folder}/finetuning/deberta/{get_name()}-fe-2"
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        trainer.train()

        trainer.evaluate()

        trainer.save_model()

        return trainer.model
    else:
        return trainer.model


class DeBERTaConfig(BaseModel):
    task_dir: str = (None,)
    task_name: str = "RTD"
    data_dir: str = settings.deberta_data_dir
    output_dir: str = f"{settings.checkpoints_folder}/{get_name()}"
    max_seq_length: int = 64  # After tokenization, the max_seq_length is different
    debug: bool = False
    init_model: str = None
    vocab_path: str = settings.deberta_vocab_path  # Get from tokenizer
    vocab_type: str = "spm"  # ?
    bert_config: str = settings.deberta_config
    cls_drop_out: float = None
    tag: str = "DemainBERT"
    pre_trained: str = None
    vat_lambda: float = 0
    vat_learning_rate: float = 4e-5
    vat_init_perturbation: float = 1e-2
    vat_loss_fn: str = "symmetric-kl"
    export_onnx_model: bool = False
    warmup: int = 20000
    fp16: bool = True
    decoupled_training: bool = True
    learning_rate: float = 3e-5
    num_train_epochs: int = 1
    num_training_steps: int = settings.deberta_num_training_steps
    do_train: bool = True
    train_batch_size: int = settings.deberta_train_batch_size
    eval_batch_size: int = settings.deberta_eval_batch_size
    init_generator: str = (
        settings.deberta_pretrained_generator_path
        if settings.deberta_pretrained_generator_path
        else None
    )
    init_discriminator: str = (
        settings.deberta_pretrained_discriminator_path
        if settings.deberta_pretrained_discriminator_path
        else None
    )
    world_size: int = 1
    rank: int = 1
    master_ip: str = None
    master_port: str = None
    seed: int = 1234
    n_gpu: int = 1
    node_rank: int = 1
    local_rank: int = 1
    checkpoint_dir: str = "checkpoints/pretraining/deberta"
    discriminator_learning_rate: float = -1
    dump_interval: int = 10000
    workers: int = 2
    loss_scale: float = 16384
    scale_steps: int = 250
    lookahead_k: int = -1
    lookahead_alpha: float = 0.5
    with_radam: bool = False
    opt_type: str = "adam"
    warmup_proportion: float = 0.1
    lr_schedule_ends: float = 0
    lr_schedule: str = "warmup_linear"
    max_grad_norm: float = 1
    epsilon: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    rtd_lambda: float = 10


def pre_train():
    # Create Masked Language Model
    args = DeBERTaConfig()
    vocab_path, vocab_type = load_vocab(
        vocab_path=args.vocab_path,
        vocab_type=args.vocab_type,
        pretrained_id=args.init_model,
    )
    tokenizer = tokenizers[vocab_type](vocab_path)
    task = get_task("RTD")(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        data_dir=args.data_dir,
        args=args,
    )
    label_list = task.get_labels()

    eval_data = task.eval_data(max_seq_len=args.max_seq_length)
    logger.info("  Evaluation batch size = %d", args.eval_batch_size)

    train_data = task.train_data(max_seq_len=args.max_seq_length, debug=args.debug)
    model_class_fn = task.get_model_class_fn()
    model = create_model(args, len(label_list), model_class_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8"
    ) as fs:
        fs.write(model.config.to_json_string() + "\n")
    shutil.copy(vocab_path, args.output_dir)
    logger.info("Model config {}".format(model.config))
    device = initialize_distributed(args)
    if not isinstance(device, torch.device):
        return 0
    model.to(device)
    run_eval_fn = task.get_eval_fn()
    loss_fn = task.get_loss_fn(args)
    if run_eval_fn is None:
        run_eval_fn = run_eval

    train_fn = task.get_train_fn(args, model)
    pretrain_deberta_model(
        args,
        model,
        device,
        train_data,
        eval_data,
        run_eval_fn,
        loss_fn=loss_fn,
        train_fn=train_fn,
    )

