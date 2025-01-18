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

import logging
import time

import numpy as np
import pandas as pd

from dga_classifier import __get_all_idgas, get_name, save_metrics, set_seed, get_settings
from dga_classifier.trainer import run

logger = logging.getLogger("dga_classifier.experiments")
settings = get_settings()


def run_balanced_idn_analysis():
    # Run 5 fold-cross validation training on everything available unbalanced
    # When evaluating, take notice of benign languages and how they are evaluated
    logger.info("\n\n\n #### Starting Balanced IDN Analysis ####")
    logger.info("\n\n")
    result = {}

    idgas = __get_all_idgas(exclude_scripts=True)

    logger.info(f"Train on {idgas}")
    result["no_idn_training"] = run()
    result["balanced"] = run(
        include_idga=idgas,
        balanced_idn_training=True,
    )
    result["unbalanced"] = run(include_idga=idgas)
    name = get_name()
    save_metrics(name, result)


def run_real_world_evaluation():
    logger.info("\n\n\n #### Starting Real World Study ####")
    logger.info("\n\n")
    for i in range(5):
        logger.info(f"Executing real world run {i}")
        seed = 43 + i
        set_seed(seed)
        logger.info(f"Setting seed to: {seed}")
        result = run(rw_iteration=i)
        name = get_name()
        save_metrics(name, result, folder="rw")


def run_idga_evaluation():
    result = run(evaluate_idgas=True)[0]
    result["idga"] = "No IDN Trained"
    df = pd.DataFrame([result])
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"evaluations/idga/{formatted_name}", index=False)


def run_idga_trained_evaluation():
    all_idgas = __get_all_idgas()
    result = run(evaluate_idgas=True, include_idga=all_idgas)[0]
    result["idga"] = "IDN Trained"
    df = pd.DataFrame([result])
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"evaluations/idga/{formatted_name}", index=False)


def run_include_only_one_script_analysis():
    # Include only a single script
    results = []
    idgas = __get_all_idgas(exclude_languages=True)
    scripts = [
        "Latin",
        "Han",
        "Common",
        "Katakana",
        "Thai",
        "Hangul",
        "Hiragana",
        "Cyrillic",
        "Arabic",
        "Hebrew",
        "Devanagari",
        "Tifinagh",
        "Buginese",
        "Bamum",
    ]

    script_idgas = [
        "charbot",
        "corebot",
        "dircrypt",
        "qadars",
    ]

    for script in scripts:

        train_on_idgas = []

        # Get all idgas except of one language
        for idga in idgas:
            script_split = idga.split("-")
            idga_name = script_split[0]
            # If not a language parameterized idga, add to training
            if idga_name not in script_idgas:
                continue
            # Get script of idga
            idga_script = script_split[1].split(":")[1]
            # If language is not to be excluded, add to training
            if script in idga_script:
                train_on_idgas.append(idga)

        logger.info(f"Including {script} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]

        result["trained_on_script"] = f"{script}"
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"evaluations/idga/{formatted_name}", index=False)


def run_leave_one_script_out_analysis():
    # Leave the script including the languages of the script out
    results = []
    idgas = __get_all_idgas(exclude_languages=True)
    scripts = [
        "Latin",
        "Han",
        "Common",
        "Katakana",
        "Thai",
        "Hangul",
        "Hiragana",
        "Cyrillic",
        "Arabic",
        "Hebrew",
        "Devanagari",
        "Tifinagh",
        "Buginese",
        "Bamum",
    ]

    script_idgas = [
        "charbot",
        "corebot",
        "dircrypt",
        "qadars",
    ]

    for script in scripts:

        train_on_idgas = []

        # Get all idgas except of one language
        for idga in idgas:

            script_split = idga.split("-")
            idga_name = script_split[0]
            # If not a language parameterized idga, add to training
            if idga_name not in script_idgas:
                train_on_idgas.append(idga)
                continue
            # Get language of idga
            idga_script = script_split[1].split(":")[1]
            # If language is not to be excluded, add to training
            if script not in idga_script:
                train_on_idgas.append(idga)

        logger.info(f"Excluding {script} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]

        result["left_out_language"] = f"{script}"
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"evaluations/idga/{formatted_name}", index=False)


def run_include_only_one_language_analysis():
    results = []
    idgas = __get_all_idgas(exclude_scripts=True)
    languages = [
        "ko",
        "ja",
        "es",
        "th",
        "tr",
        "de",
        "fr",
        "sv",
        "zh",
        "ru",
        "mr",
        "ur",
        "hi",
        "ar",
        "vi",
        "ml",
        "te",
    ]

    lang_idgas = [
        "charbot",
        "corebot",
        "dircrypt",
        "gozi",
        "nymaim2",
        "suppobox",
        "qadars",
    ]

    for lang in languages:

        train_on_idgas = []

        # Get all idgas except of one language
        for idga in idgas:

            lang_split = idga.split("-")
            idga_name = lang_split[0]
            # If not a language parameterized idga, add to training
            if idga_name not in lang_idgas:
                continue
            # Get language of idga
            idga_language = lang_split[1].split(":")[1]
            # If language is not to be excluded, add to training
            if lang in idga_language:
                train_on_idgas.append(idga)

        logger.info(f"Including {lang} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]

        result["trained_on_language"] = f"{lang}"
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"evaluations/idga/{formatted_name}", index=False)


def run_leave_one_language_out_analysis():
    results = []
    idgas = __get_all_idgas(exclude_scripts=True)
    languages = [
        "ko",
        "ja",
        "es",
        "th",
        "tr",
        "de",
        "fr",
        "sv",
        "zh",
        "ru",
        "mr",
        "ur",
        "hi",
        "ar",
        "vi",
        "ml",
        "te",
    ]

    lang_idgas = [
        "charbot",
        "corebot",
        "dircrypt",
        "gozi",
        "nymaim2",
        "suppobox",
        "qadars",
    ]

    for lang in languages:

        train_on_idgas = []

        # Get all idgas except of one language
        for idga in idgas:

            lang_split = idga.split("-")
            idga_name = lang_split[0]
            # If not a language parameterized idga, add to training
            if idga_name not in lang_idgas:
                train_on_idgas.append(idga)
                continue
            # Get language of idga
            idga_language = lang_split[1].split(":")[1]
            # If language is not to be excluded, add to training
            if lang not in idga_language:
                train_on_idgas.append(idga)

        logger.info(f"Excluding {lang} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]

        result["left_out_language"] = f"{lang}"
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"{settings.evaluation_folder}/idga/{formatted_name}", index=False)


def run_include_only_one_idga_analysis():
    results = []
    all_idgas_variants = __get_all_idgas(exclude_scripts=False)
    all_idgas = set([idga.split("-")[0] for idga in all_idgas_variants])

    for include_idga in all_idgas:
        train_on_idgas = [
            idga for idga in all_idgas_variants if idga.startswith(include_idga)
        ]
        logger.info(f"Including {include_idga} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]
        result["left_out_idga"] = include_idga
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"{settings.evaluation_folder}/idga/{formatted_name}", index=False)


def run_leave_one_idga_out_analysis():
    results = []
    all_idgas_variants = __get_all_idgas(exclude_scripts=False)
    all_idgas = set([idga.split("-")[0] for idga in all_idgas_variants])

    for except_idga in all_idgas:
        train_on_idgas = [
            idga for idga in all_idgas_variants if not idga.startswith(except_idga)
        ]
        logger.info(f"Excluding {except_idga} -> Train on {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]
        result["left_out_idga"] = except_idga
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(f"{settings.evaluation_folder}/idga/{formatted_name}", index=False)


def run_include_only_one_idga_variant_analysis():
    results = []
    idgas = __get_all_idgas(exclude_scripts=False)
    idgas.sort()
    parts = np.linspace(0, len(idgas), 21, endpoint=True, dtype=int)
    selected_range = range(parts[settings.iooiv_index], parts[settings.iooiv_index + 1])
    logger.info("Selected range", selected_range)
    for i in selected_range:
        idga = idgas[i]
        logger.info(f"Training includes IDGA {idga}")
        result = run(
            include_idga=[idga],
            evaluate_idgas=True,
            single_fold=True,
        )[0]
        result["idga"] = f"{idga}-trained"
        results.append(result)
    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(
        f"{settings.evaluation_folder}/idga/{settings.iooiv_index}-{formatted_name}",
        index=False,
    )


def run_leave_one_idga_variant_out_analysis():
    results = []
    idgas = __get_all_idgas(exclude_scripts=False)
    idgas.sort()
    parts = np.linspace(0, len(idgas), settings.loivo_split_n, endpoint=True, dtype=int)
    selected_range = range(parts[settings.loivo_index], parts[settings.loivo_index + 1])
    logger.info("LOIVO INDEX:", settings.loivo_index)
    logger.info("Selected range:", selected_range)
    logger.info("Range includes IDGAS:", [idgas[i] for i in selected_range])
    for i in selected_range:
        left_out_idga = idgas[i]
        logger.info(f"Leave {left_out_idga} out")
        train_on_idgas = [idga for idga in idgas if idga != left_out_idga]
        # train_on_idgas = idgas[:i] + idgas[i + 1 :]
        logger.info(f"Train on IDGAS: {train_on_idgas}")
        result = run(
            include_idga=train_on_idgas,
            evaluate_idgas=True,
            single_fold=True,
        )[0]
        result["left_out_idga"] = f"{left_out_idga}-left-out"
        results.append(result)

    df = pd.DataFrame(results)
    formatted_name = time.strftime("%Y%m%d-%H%M%S") + "_" + get_name() + ".csv"
    df.to_csv(
        f"{settings.evaluation_folder}/idga/{settings.loivo_index}-{formatted_name}",
        index=False,
    )


def run_statistical_analysis():
    logger.info("\n\n\n #### Starting Statistical Analysis ####")
    logger.info("\n\n")
    for i in range(10):
        seed = 43 + i
        set_seed(seed)
        logger.info(f"Setting seed to: {seed}")
        result = run()
        name = get_name()
        name += f"-{seed}"
        logger.info("Saving run")
        save_metrics(name, result, folder="metrics/statistical_analysis/")
        logger.info("\n\n")
