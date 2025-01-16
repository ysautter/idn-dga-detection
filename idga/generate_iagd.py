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

import importlib
import os
import sys
from functools import partial

import pandas as pd

# Parameters
csv_file = "agid.csv"
threshold = 2000

# Load the CSV file
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(
            columns=["agid", "type"]
        )  # Assuming the CSV has the flag column


# Save the DataFrame to the CSV file
def save_csv(df, file_path):
    df.to_csv(file_path, index=False)


# Generate additional rows
def generate_rows(num_rows, idga_specifier):
    additional_args = None
    res = idga_specifier.split("-")
    idga = res[0]
    idga_character_specifier = None
    if len(res) == 2:
        idga_character_specifier = res[1]
        res = idga_character_specifier.split("|")
        idga_character_specifier = res[0]
        if len(res) == 2:
            additional_args = res[1]

    idga_module = importlib.import_module(
        f"international_domain_generation_algorithms.{idga}.dga"
    )
    if additional_args is not None:
        idga_domains = idga_module.gen_iagds(
            num_rows, idga_character_specifier, additional_args
        )
    else:
        idga_domains = idga_module.gen_iagds(num_rows, idga_character_specifier)
    new_rows = pd.DataFrame(
        {"type": [idga_specifier] * len(idga_domains), "agid": idga_domains}
    )
    return new_rows


# Check the number of rows with the specific flag and add more if needed
def check_and_add_rows(df, column, value, min_rows):
    current_count = df[df[column] == value].shape[0]
    if current_count < min_rows:
        try:
            additional_rows = generate_rows(min_rows - current_count, value)
            df = pd.concat([df, additional_rows], ignore_index=True)
        except FileNotFoundError as e:
            print(f"Generate files for {value}")

    return df


# Handler for unexpected exit or crash
def handle_exit(df):
    save_csv(df, csv_file)
    sys.exit(0)


def main():
    top_languages = [
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
        "pt",
        "hu",
    ]
    least_languages = ["ml", "kn", "te", "pa", "gu", "ta", "bn", "ne", "mr", "ur", "hi"]
    top_scripts = [
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
        "Greek",
        "Khmer",
    ]
    least_scripts = [
        "Hanunoo",
        "Vai",
        "Mongolian",
        "Nko",
        "Cuneiform",
        "Glagolitic",
        "Yi",
        "New_Tai_Lue",
        "Tifinagh",
        "Buginese",
        "Bamum",
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

    script_idgas = [
        "charbot",
        "corebot",
        "dircrypt",
        "qadars",
    ]

    additional_languages = ["ar", "vi", "ml", "te"]
    print(top_languages[:10])
    print(least_languages[-3:])
    print(additional_languages)

    l_idgas = []
    for idga in lang_idgas:
        for lang in top_languages[:10]:
            if idga == "suppobox":
                l_idgas.extend([f"{idga}-lang:{lang}|{i+1}" for i in range(3)])
            else:
                l_idgas.append(f"{idga}-lang:{lang}")

    for idga in lang_idgas:
        for lang in additional_languages:
            if idga == "suppobox":
                l_idgas.extend([f"{idga}-lang:{lang}|{i+1}" for i in range(3)])
            else:
                l_idgas.append(f"{idga}-lang:{lang}")

    for idga in lang_idgas:
        for lang in least_languages[-3:]:
            if idga == "suppobox":
                l_idgas.append(f"{idga}-lang:{lang}|1")
                l_idgas.append(f"{idga}-lang:{lang}|2")
            else:
                l_idgas.append(f"{idga}-lang:{lang}")

    s_idgas = [
        f"{idga}-script:{script}"
        for idga in script_idgas
        for script in top_scripts[:10]
    ]

    s_idgas.extend(
        [
            f"{idga}-script:{script}"
            for idga in script_idgas
            for script in least_scripts[-3:]
        ]
    )

    idgas = []

    idgas.extend(
        [
            "orchard",
            "pushdo",
            "sharkbot",
        ],
    )
    idgas += l_idgas
    idgas += s_idgas
    # Load the CSV file
    df = load_csv(csv_file)

    print(len(idgas))
    print(idgas)

    for idga in idgas:
        df = check_and_add_rows(df, "type", idga, threshold)
        df.drop_duplicates(subset=["agid"])
        save_csv(df, csv_file)

    save_csv(df, csv_file)


if __name__ == "__main__":
    main()
