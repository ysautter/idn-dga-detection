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

import json
import os
import time

import random
import evaluate
import numpy as np
import torch
import pandas as pd

from functools import lru_cache

from pydantic import ValidationError
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils import check_random_state

from dga_classifier.config import Settings

target_names = [
    "benign",
    "orchardgenesis",
    "emotet",
    "qakbot",
    "omexo",
    "pushdo",
    "urlzone",
    "goznym",
    "fobber",
    "padcrypt",
    "monerominer",
    "m0yvtdd",
    "ares",
    "randomloader",
    "sharkbot",
    "phorpiex",
    "mirai",
    "pitou",
    "ud2",
    "vidro",
    "conficker",
    "alien",
    "makloader",
    "corebot",
    "gameover",
    "suppobox",
    "nymaim",
    "tempedrevetdd",
    "flubot",
    "nymaim2",
    "dnschanger",
    "darkwatchman",
    "m0yv",
    "beebone",
    "sphinx",
    "g01",
    "dyre",
    "pushdotid",
    "xxhex",
    "ekforward",
    "ramdo",
    "sutra",
    "mydoom",
    "chinad",
    "virut",
    "bedep",
    "dmsniff",
    "locky",
    "banjori",
    "symmi",
    "pykspa",
    "darkshell",
    "modpack",
    "ud3",
    "proslikefan",
    "infy",
    "diamondfox",
    "hesperbot",
    "qhost",
    "pykspa2s",
    "feodo",
    "ebury",
    "madmax",
    "kfos",
    "vawtrak",
    "downloader",
    "pandabanker",
    "gozi",
    "bazarloader",
    "wd",
    "blackhole",
    "xshellghost",
    "bobax",
    "tofsee",
    "ud5",
    "dircrypt",
    "bamital",
    "necro",
    "vidrotid",
    "chir",
    "matsnu",
    "sisron",
    "kingminer",
    "murofetweekly",
    "necurs",
    "chaes",
    "ramnit",
    "rovnix",
    "enviserv",
    "ccleaner",
    "gspy",
    "pykspa2",
    "pseudomanuscrypt",
    "oderoor",
    "tsifiri",
    "simda",
    "tinynuke",
    "torpig",
    "bigviktor",
    "qsnatch",
    "zloader",
    "shifu",
    "szribi",
    "tinba",
    "qadars",
    "metastealer",
    "redyms",
    "gameover_p2p_a37c4dbb",
    "orchard",
    "dnsbenchmark",
    "murofet",
    "cryptolocker",
    "tinyfluff",
    "tempedreve",
    "volatilecedar",
    "ranbyus",
]


@lru_cache()
def get_settings():
    return Settings()


try:
    settings = get_settings()
except ValidationError as e:
    for error in e.errors():
        print(error["msg"])
    exit(1)

evaluation_folder = settings.evaluation_folder


def save_metrics(name, data, folder="metrics"):

    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)

    formatted_name = (
        time.strftime("%Y%m%d-%H%M%S") + "_" + name.replace(" ", "_").lower() + ".json"
    )
    save_folder = f"{evaluation_folder}/{folder}"
    os.makedirs(
        save_folder,
        exist_ok=True,
    )
    output_filename = os.path.join(save_folder, formatted_name)
    with open(output_filename, "w") as file:
        json.dump(data, file, indent=4)


def get_name():
    training = "Finetune"
    if settings.pre_training:
        training = "Pretrain"
    if settings.bert_size != "base":
        return f"{settings.model}-{settings.bert_size}-{settings.classification_task}-{settings.project_name}-{training}"
    return f"{settings.model}-{settings.classification_task}-{settings.project_name}-{training}"


def calculate_metrics(
    labels, prediction_logits, save: bool = True,
):
    name = get_name()
    data = {
        "name": name,
        "predictions": prediction_logits.tolist(),
        "labels": labels.tolist(),
    }
    if settings.classification_task == "multiclass":
        predictions = torch.argmax(prediction_logits, 1)
        precision_macro = precision_score(labels, predictions, average="macro")
        recall_macro = recall_score(labels, predictions, average="macro")
        f1_macro = f1_score(labels, predictions, average="macro")
        data.update(
            {
                "f1-score": f1_macro,
                "precision": precision_macro,
                "recall": recall_macro,
                "classification_report": classification_report(
                    labels,
                    predictions,
                    digits=4,
                    labels=range(len(target_names)),
                    target_names=target_names,
                    output_dict=True,
                ),
            }
        )

    if settings.classification_task == "binary":
        prediction_probs = torch.sigmoid(prediction_logits)[:, 1]

        fpr, tpr, _ = roc_curve(labels, prediction_probs)

        accuracy = evaluate.load("accuracy", experiment_id=get_name())
        predictions = prediction_probs >= 0.5
        acc = accuracy.compute(predictions=predictions, references=labels)
        try:
            tpr_at_fpr_0001 = tpr[fpr <= 0.001][-1]
        except IndexError:
            tpr_at_fpr_0001 = 0
        try:
            tpr_at_fpr_001 = tpr[fpr <= 0.01][-1]
        except IndexError:
            tpr_at_fpr_001 = 0
        try:
            tpr_at_fpr_01 = tpr[fpr <= 0.1][-1]
        except IndexError:
            tpr_at_fpr_01 = 0
        try:
            roc_auc_score_at_max_fpr_0001 = roc_auc_score(
                labels, prediction_probs, max_fpr=0.001
            )
        except ValueError:
            roc_auc_score_at_max_fpr_0001 = 0
        try:
            roc_auc_score_at_max_fpr_001 = roc_auc_score(
                labels, prediction_probs, max_fpr=0.01
            )
        except ValueError:
            roc_auc_score_at_max_fpr_001 = 0
        try:
            roc_auc_score_at_max_fpr_01 = roc_auc_score(
                labels, prediction_probs, max_fpr=0.1
            )
        except ValueError:
            roc_auc_score_at_max_fpr_01 = 0
        data.update(
            {
                "acc": acc["accuracy"],
                "tpr@fpr=0.001": tpr_at_fpr_0001,
                "tpr@fpr=0.01": tpr_at_fpr_001,
                "tpr@fpr=0.1": tpr_at_fpr_01,
                "roc_auc@max_fpr=0.001": roc_auc_score_at_max_fpr_0001,
                "roc_auc@max_fpr=0.01": roc_auc_score_at_max_fpr_001,
                "roc_auc@max_fpr=0.1": roc_auc_score_at_max_fpr_01,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }
        )

    if save:
        save_metrics(name, data)


    return data


def __get_all_idgas(exclude_scripts: bool = False, exclude_languages: bool = False):
    # Get all idgas
    df = pd.read_csv(settings.idn_data_mal, header=None)
    idgas = list(set(df.iloc[:, 1]))
    if exclude_scripts:
        idgas = [idga for idga in idgas if "script" not in idga]
    if exclude_languages:
        idgas = [idga for idga in idgas if "lang" not in idga]
    return idgas


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    check_random_state(seed)

