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

import multiprocessing
import random
import string
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm

from dga_classifier import get_settings

model_max_length = 63  # 253 + start and end token?
chars = string.ascii_lowercase + string.digits + "-" + "_" + "."

char_to_int = {ch: i + 1 for i, ch in enumerate(chars)}
int_to_char = {v: k for k, v in char_to_int.items()}
label_to_int = {
    "benign": 0,
    "orchardgenesis": 1,
    "emotet": 2,
    "qakbot": 3,
    "omexo": 4,
    "pushdo": 5,
    "urlzone": 6,
    "goznym": 7,
    "fobber": 8,
    "padcrypt": 9,
    "monerominer": 10,
    "m0yvtdd": 11,
    "ares": 12,
    "randomloader": 13,
    "sharkbot": 14,
    "phorpiex": 15,
    "mirai": 16,
    "pitou": 17,
    "ud2": 18,
    "vidro": 19,
    "conficker": 20,
    "alien": 21,
    "makloader": 22,
    "corebot": 23,
    "gameover": 24,
    "suppobox": 25,
    "nymaim": 26,
    "tempedrevetdd": 27,
    "flubot": 28,
    "nymaim2": 29,
    "dnschanger": 30,
    "darkwatchman": 31,
    "m0yv": 32,
    "beebone": 33,
    "sphinx": 34,
    "g01": 35,
    "dyre": 36,
    "pushdotid": 37,
    "xxhex": 38,
    "ekforward": 39,
    "ramdo": 40,
    "sutra": 41,
    "mydoom": 42,
    "chinad": 43,
    "virut": 44,
    "bedep": 45,
    "dmsniff": 46,
    "locky": 47,
    "banjori": 48,
    "symmi": 49,
    "pykspa": 50,
    "darkshell": 51,
    "modpack": 52,
    "ud3": 53,
    "proslikefan": 54,
    "infy": 55,
    "diamondfox": 56,
    "hesperbot": 57,
    "qhost": 58,
    "pykspa2s": 59,
    "feodo": 60,
    "ebury": 61,
    "madmax": 62,
    "kfos": 63,
    "vawtrak": 64,
    "downloader": 65,
    "pandabanker": 66,
    "gozi": 67,
    "bazarloader": 68,
    "wd": 69,
    "blackhole": 70,
    "xshellghost": 71,
    "bobax": 72,
    "tofsee": 73,
    "ud5": 74,
    "dircrypt": 75,
    "bamital": 76,
    "necro": 77,
    "vidrotid": 78,
    "chir": 79,
    "matsnu": 80,
    "sisron": 81,
    "kingminer": 82,
    "murofetweekly": 83,
    "necurs": 84,
    "chaes": 85,
    "ramnit": 86,
    "rovnix": 87,
    "enviserv": 88,
    "ccleaner": 89,
    "gspy": 90,
    "pykspa2": 91,
    "pseudomanuscrypt": 92,
    "oderoor": 93,
    "tsifiri": 94,
    "simda": 95,
    "tinynuke": 96,
    "torpig": 97,
    "bigviktor": 98,
    "qsnatch": 99,
    "zloader": 100,
    "shifu": 101,
    "szribi": 102,
    "tinba": 103,
    "qadars": 104,
    "metastealer": 105,
    "redyms": 106,
    "gameover_p2p": 107,
    "orchard": 108,
    "dnsbenchmark": 109,
    "murofet": 110,
    "cryptolocker": 111,
    "tinyfluff": 112,
    "tempedreve": 113,
    "volatilecedar": 114,
    "ranbyus": 115,
    "ud4": 116,
}

rw_eval_dgas = [
    "bamital",
    "ccleaner",
    "chinad",
    "conficker",
    "corebot",
    "cryptolocker",
    "diamondfox",
    "dyre",
    "ekforward",
    "emotet",
    "gameover",
    "gameover_p2p",
    "gozi",
    "infy",
    "locky",
    "madmax",
    "matsnu",
    "modpack",
    "murofet",
    "murofetweekly",
    "necurs",
    "nymaim",
    "nymaim2",
    "oderoor",
    "padcrypt",
    "pandabanker",
    "proslikefan",
    "pushdo",
    "pykspa",
    "qadars",
    "qakbot",
    "ranbyus",
    "sisron",
    "sphinx",
    "suppobox",
    "symmi",
    "szribi",
    "tempedrevetdd",
    "tinynuke",
    "tofsee",
    "torpig",
    "ud2",
    "vidro",
    "virut",
    "wd",
    "xshellghost",
]


target_labels = [None] * len(label_to_int)
for key, index in label_to_int.items():
    target_labels[index] = key

num_proc = multiprocessing.cpu_count()
settings = get_settings()

# IDN datasets
IDN_DATA_MAL = settings.idn_data_mal
IDN_DATA_BENIGN = settings.idn_data_benign

# Test split for balanced language evaluation
IDN_DATA_BENIGN_TEST = settings.idn_data_benign_test
IDN_DATA_MAL_TEST = settings.idn_data_mal_test

# Original balanced benign and malicious domain datasets
BENIGN_DATASET = settings.benign_dataset
MAL_DATASET = settings.mal_dataset

# Real-World train and eval dataset
BENIGN_RW_TEST_DATASET = settings.benign_rw_test_dataset
MAL_RW_TEST_DATASET = settings.mal_rw_test_dataset

BENIGN_RW_DATASET = settings.benign_rw_dataset
MAL_RW_DATASET = settings.mal_rw_dataset

# Multiclass datasets
MAL_MULTICLASS_DATASET = settings.mal_multiclass_dataset
BENIGN_MULTICLASS_DATASET = settings.benign_multiclass_dataset


def preprocess_data(data: List, tokenizer=None):
    processed_data = []
    if tokenizer:
        for element in tqdm(data):
            element = element.lower()
            processed_element = tokenizer(
                element, padding="max_length", max_length=model_max_length
            )
            processed_data.append(processed_element.data)
    else:
        for element in tqdm(data):
            element = element.lower()
            mapping = list(map(char_to_int.get, element))
            length = model_max_length - len(mapping)
            padded = np.pad(mapping, (0, length)).tolist()
            processed_element = {
                "input_ids": padded,
                "attention_mask": [el if el == 0 else 1 for el in padded],
                "token_type_ids": [0] * len(padded),
            }
            processed_data.append(processed_element)
    return processed_data


def get_dataset(domains, orig_labels, tokenization, tokenizer=None):
    if settings.classification_task == "binary":
        labels = [1 if label > 0 else 0 for label in orig_labels]
    else:
        labels = orig_labels
    if not isinstance(domains, list):
        domains = domains.tolist()
    dataset = datasets.Dataset.from_dict(
        {"domain": domains, "label": labels, "orig_label": orig_labels}
    )

    dataset = dataset.with_format("torch")
    dataset = dataset.map(tokenization, batched=tokenizer is not None)

    dataset.set_format()

    del domains, labels
    return dataset


def get_datasets(
    train_domains, train_labels, eval_domains, eval_labels, tokenization, tokenizer=None
):
    return get_dataset(
        train_domains, train_labels, tokenization, tokenizer
    ), get_dataset(eval_domains, eval_labels, tokenization, tokenizer)


def get_data(
    preprocess: bool = True,
    tokenizer=None,
    include_idga: List[str] = None,
    evaluate_rw: bool = False,
    rw_iteration: int = 0,
    balanced_idn_training: bool = False,
    evaluate_balanced_idn: bool = False,
):
    if evaluate_rw:
        mal = pd.read_csv(MAL_RW_TEST_DATASET, header=None)
        benign = pd.read_csv(BENIGN_RW_TEST_DATASET, header=None)
    elif evaluate_balanced_idn:
        mal = pd.read_csv(IDN_DATA_MAL_TEST, header=None)
        mal.columns = ["domain", "label"]
        mal["lang"] = mal["label"].apply(get_lang_from_idga_variant)

        benign = pd.read_csv(IDN_DATA_BENIGN_TEST, header=None)
        benign.columns = ["domain", "label", "lang"]
        benign["lang"] = benign["lang"].replace("zh-cn", "zh")
    else:
        if settings.classification_task == "binary":
            if settings.real_world_study:
                mal = pd.read_csv(
                    MAL_RW_DATASET.replace("x", str(rw_iteration)), header=None
                )
                benign = pd.read_csv(BENIGN_RW_DATASET, header=None)
            else:
                mal = pd.read_csv(MAL_DATASET, header=None)
                benign = pd.read_csv(BENIGN_DATASET, header=None)
        else:
            mal = pd.read_csv(MAL_MULTICLASS_DATASET, header=None)
            benign = pd.read_csv(BENIGN_MULTICLASS_DATASET, header=None)

    data = pd.concat([benign, mal], ignore_index=True).dropna()

    # Shuffling data
    data = data.sample(frac=1).reset_index(drop=True)

    domains = data.iloc[:, 0]
    labels_strings = data.iloc[:, 1].map(lambda x: x.split("_dga_")[0])
    label_names = set(labels_strings)
    label_names.discard("benign")

    labels = []
    for label in labels_strings:
        if label == "gameover_p2p_a37c4dbb":
            labels.append(label_to_int["gameover_p2p"])
            continue
        if label == "blocklist":
            labels.append(1000)
            continue
        labels.append(label_to_int[label])

    domains = domains.tolist()

    if include_idga is not None:
        idn_domains, idn_labels = get_idn_data(
            idga=include_idga, balanced_idn_training=balanced_idn_training
        )
        highest_label = len(set(labels_strings)) + 1
        idn_labels = [
            label + highest_label if label > 0 else label
            for i, label in enumerate(idn_labels)
        ]
        domains.extend(idn_domains)
        labels.extend(idn_labels)
        combined = list(zip(domains, labels))
        random.shuffle(combined)
        domains, labels = zip(*combined)

    if preprocess:
        texts_processed = preprocess_data(domains, tokenizer)
    else:
        texts_processed = domains

    return texts_processed, labels


def get_lang_from_idga_variant(variant):
    split = variant.split(":")
    if len(split) > 1:
        split = split[1].split("|")[0]
    else:
        split = split[0]
    return split


def get_idn_data(
    idga: List[str] = None,
    get_only_mal: bool = False,
    balanced_idn_training: bool = False,
):
    mal = pd.read_csv(IDN_DATA_MAL, header=None)
    mal.columns = ["domain", "label"]
    mal["lang"] = mal["label"].apply(get_lang_from_idga_variant)

    benign = pd.read_csv(IDN_DATA_BENIGN, header=None)

    benign.columns = ["domain", "label", "lang"]
    benign["lang"] = benign["lang"].replace("zh-cn", "zh")

    if idga is not None:
        mal = mal[mal["label"].isin(idga)]
        if balanced_idn_training:
            language_counts = benign["lang"].value_counts()
            language_counts["orchard"] = 200
            language_counts["pushdo"] = 200
            language_counts["sharkbot"] = 200
            sampled_mal_df = (
                mal.groupby("lang")
                .apply(lambda x: x.sample(n=language_counts[x.name], replace=True))
                .reset_index(drop=True)
            )
            sampled_mal_df = (
                sampled_mal_df.groupby(["label", "lang"])
                .apply(lambda x: x.sample(frac=1))
                .reset_index(drop=True)
            )
            mal = sampled_mal_df
            benign = benign[benign["lang"].isin(mal["lang"].unique())]

        elif len(mal) > len(benign):
            sample_size = len(benign) // len(idga)
            sample_size = min(sample_size, mal.groupby("label").size().min())
            print(f"Using {sample_size} AGIDs from IDGAs")
            # sample_size = 1550
            mal = (
                mal.groupby("label")
                .apply(lambda x: x.sample(sample_size))
                .reset_index(drop=True)
            )

            benign = benign.sample(n=len(mal)).reset_index(drop=True)

    if get_only_mal:
        data = mal
    else:
        data = pd.concat([benign, mal], ignore_index=True).dropna()

    # Shuffling data
    data = data.sample(frac=1).reset_index(drop=True)

    domains = data["domain"]
    labels = data["label"]
    langs = data["lang"]
    label_names = set(labels)
    label_names.discard("benign")
    label_to_int = {label: idx + 1 for idx, label in enumerate(set(label_names))}

    # The benign IDNS are assigned to a negative label, such that we can perform stratified sampling
    # such that we use the same amount of benign IDN per language in each fold
    label_to_int["benign"] = 0
    labels = [label_to_int[label] for label in labels]

    new_benign_labeled_labels = []
    lang_to_int = {lang: -idx - 1 for idx, lang in enumerate(langs.unique())}
    for domain, label, lang in zip(domains, labels, langs):
        if label != 0:
            new_benign_labeled_labels.append(label)
            continue
        new_benign_labeled_labels.append(lang_to_int[lang])

    texts_processed = domains.tolist()

    return texts_processed, new_benign_labeled_labels


def get_train_and_eval_dataset(tokenizer):
    mal = pd.read_csv("data/mal.csv", header=None)
    benign = pd.read_csv("data/benign.csv", header=None)

    data = pd.concat([benign, mal], ignore_index=True).dropna()

    # Shuffling data
    data = data.sample(frac=1).reset_index(drop=True)

    domains = data.iloc[:, 0]
    labels = data.iloc[:, 1].map(lambda x: 0 if x == "benign" else 1)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        domains, labels, test_size=0.2
    )

    train_texts_processed = preprocess_data(train_texts.tolist())
    val_texts_processed = preprocess_data(val_texts.tolist())

    train_dataset = DGADataset(train_texts_processed, train_labels.tolist())
    eval_dataset = DGADataset(val_texts_processed, val_labels.tolist())

    return train_dataset, eval_dataset


class DGADataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels = labels
        self.return_binary = True

    def set_binary(self):
        self.return_binary = True

    def set_multilabel(self):
        self.return_binary = False

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.return_binary:
            # labels = labels.map(lambda x: 1 if x > 0 else 0)
            label = 1 if label > 0 else 0
        item = {
            "input_ids": torch.tensor(self.domains[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.domains[idx]["attention_mask"]),
            "labels": torch.tensor(label),
        }
        return item

    def __len__(self):
        return len(self.labels)
