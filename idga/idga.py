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
import random
import re
import time
import unicodedata
import idna
from typing import Tuple
import homoglyphs as hg
import translators as ts

import numpy as np
import requests
from idna import uts46_remap, idnadata, intranges_contain

script_file_path = os.path.join(os.path.dirname(__file__), "Scripts.txt")
cldr_path = os.path.join(os.path.dirname(__file__), "cldr/cldr-misc-full/main")

dnscharset = (
    [chr(x) for x in range(0x61, 0x61 + 26)]
    + [chr(x) for x in range(0x30, 0x30 + 10)]
    + ["-"]
)


def parse_scripts_file(file_path):
    # Dictionary to store the script ranges
    script_ranges = {}

    # Regular expression to match lines with code ranges
    pattern = re.compile(r"^([0-9A-F]{4,6})(?:..([0-9A-F]{4,6}))?\s+;\s+(\w+)\s+#")

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if match:
                start_code = match.group(1)
                end_code = match.group(2) if match.group(2) else start_code
                script_name = match.group(3)

                # Convert hexadecimal codes to integers
                start_code = int(start_code, 16)
                end_code = int(end_code, 16)

                # Append the range to the corresponding script name in the dictionary
                if script_name not in script_ranges:
                    script_ranges[script_name] = []
                script_ranges[script_name].append((start_code, end_code))

    return script_ranges


script_ranges = parse_scripts_file(script_file_path)


# def get_random_unicode(length):
#
#     get_char = chr
#
#     # Update this to include code point ranges to be sampled
#     include_ranges = [
#         # (0x0020, 0x007F), # Basic Latin ASCII
#         # (0x00A1, 0x00AC), # Latin-1 Supplements
#         # (0x00AE, 0x00FF), # Latin-1 Supplements
#         # (0x0100, 0x017F), # Latin Extended-A
#         # (0x0180, 0x024F), # Latin Extended-B
#         # (0x0250, 0x02AF), # IPA Extensions
#         # (0x02B0, 0x02FF), # Spacing Modifiers
#         # (0x0300, 0x036F), # Combining Diacritical Marks
#         # (0x0370, 0x0377), # Greek/Coptic Start
#         # (0x037A, 0x037E),
#         # (0x0384, 0x038A),
#         # (0x038C, 0x038C),
#         # (0x038E, 0x03FF), # Greek/Coptic End
#         # (0x0400, 0x04FF), # Cyrillic
#         # (0x0500, 0x052F), # Cyrillic Supplements
#         # (0x0531, 0x0556), # Armenian Start
#         # (0x0559, 0x058A),
#         # (0x058D, 0x058F), # Armenian End
#         # (0x0591, 0x05C7),  # Hebrew Start
#         # (0x05D0, 0x05EA),
#         # (0x05EF, 0x05F4),  # Hebrew End
#         (0x0600, 0x060F),  # Arabic
#         # (0x0610, 0x0617),  # Arabic
#     ]
#
#     alphabet = [
#         get_char(code_point)
#         for current_range in include_ranges
#         for code_point in range(current_range[0], current_range[1] + 1)
#     ]
#     return "".join(random.choice(alphabet) for i in range(length))


def is_valid_unicode_char(char: chr) -> bool:
    try:
        # Ensure the character is not a control character or otherwise invalid
        if unicodedata.category(char).startswith("C"):
            return False
        # Check if character is PVALID according to IDNA
        if not idna.intranges_contain(
            ord(char), idna.idnadata.codepoint_classes["PVALID"]
        ):
            return False
        return True
    except ValueError:
        return False


# Generate random characters within the specified code point range
def random_unicode_char(characters: list, loc) -> str:
    while True:
        # cp = random.randint(range_start, range_end)
        # cp = sample_from_range(script)
        cp = random.choice(characters)
        if loc == 0 and unicodedata.category(cp)[0] == "M":
            continue
        if is_valid_unicode_char(cp):
            return cp


def generate_valid_unicode_label(characters: list, label_length: int) -> str:

    # Construct a random label
    label = "".join(random_unicode_char(characters, i) for i in range(label_length))

    # Validate the generated label
    # label = unicodedata.normalize("NFC", label)
    label = uts46_remap(label, std3_rules=False, transitional=False)
    try:
        idna.check_label(label)
    except idna.core.IDNAError as e:
        return generate_valid_unicode_label(characters, label_length)

    return label


def sample_from_range(ranges):
    # Flatten the list of ranges to get a cumulative list of code points
    flat_list = []
    for start, end in ranges:
        flat_list.extend(range(start, end + 1))
    # Randomly sample a code point from the flattened list
    return random.choice(flat_list)


def get_ranges(script: str = None):
    flat_list = []
    for iter_script, ranges in script_ranges.items():
        if script is not None and iter_script != script:
            continue
        ranges = script_ranges[iter_script]
        for start, end in ranges:
            flat_list.extend(range(start, end + 1))
    return flat_list


def get_charset_by_script(script: str = None):
    if script == "All":
        script = None
    characters = [chr(i) for i in get_ranges(script)]
    characters = [
        char
        for char in characters
        if intranges_contain(ord(char), idnadata.codepoint_classes["PVALID"])
    ]
    return characters


def get_char_list_from_cldr_characters(cldr_characters: str) -> list[str]:
    clean_string = re.sub(r"[\[\]]", "", cldr_characters)
    raw_characters = clean_string.split()
    characters = []
    for item in raw_characters:
        if len(item) > 1:
            characters.extend(list(item))
        else:
            characters.append(item)
    characters = [
        char
        for char in characters
        if intranges_contain(ord(char), idnadata.codepoint_classes["PVALID"])
    ]
    return characters


def get_charset_by_language(
    language: str = "de",
    include_numerical: bool = True,
    include_punctuation: bool = True,
):
    with open(f"{cldr_path}/{language}/characters.json") as f:
        characters = json.load(f)
    characters = characters["main"][language]["characters"]
    res = get_char_list_from_cldr_characters(characters["exemplarCharacters"])
    if include_numerical:
        res += get_char_list_from_cldr_characters(characters["numbers"])
    if include_punctuation:
        res += get_char_list_from_cldr_characters(characters["punctuation"])
    return res


def remove_invalid_starting_modifier(label):
    if label == "":
        return label
    char_list = list(label)
    if unicodedata.category(label[0])[0] == "M":
        return remove_invalid_starting_modifier("".join(char_list[1:]))
    return label


def remove_invalid_hypens(label):
    # Convert the string to a list of characters for easier manipulation
    char_list = list(label)
    positions = [0, -1]
    # Iterate over the positions in reverse order to avoid index shifting issues
    for pos in positions:
        # Remove the character at the specified position
        try:
            if char_list[pos] == "-":
                del char_list[pos]
        except IndexError as e:
            print(f"WTF, IndexError with: {label}")
            raise e
    label = "".join(char_list)
    if label[2:4] == "--":
        del char_list[3]
    # Convert the list back to a string
    return "".join(char_list)


def homoglyph_substitution(sld):
    error = True
    sld_orig = sld
    while error:
        try:
            sld = list(sld_orig)
            i, j = np.random.randint(0, len(sld), 2)
            i_choices = set(hg.Homoglyphs().get_combinations(sld[i]))
            i_choices = set(
                [
                    uts46_remap(x, std3_rules=False, transitional=False)
                    for x in list(i_choices)
                ]
            )
            i_choices -= set(dnscharset)
            j_choices = set(hg.Homoglyphs().get_combinations(sld[j]))
            j_choices = set(
                [
                    uts46_remap(x, std3_rules=False, transitional=False)
                    for x in list(j_choices)
                ]
            )
            j_choices -= set(dnscharset)
            sld[i] = random.choice(list(i_choices))
            sld[j] = random.choice(list(j_choices))
            sld = "".join(sld)
            label = uts46_remap(sld, std3_rules=False, transitional=False)
            idna.check_label(label)
            sld = idna.encode(label).decode()
            error = False
        except Exception as e:
            pass
    return sld


def translate_text(text, target_language="DE", translator="deepl"):
    if translator != "deepl":
        error = True
        i = 1
        while error:
            try:
                result = ts.translate_text(
                    query_text=text,
                    translator=translator,
                    from_language="en",
                    to_language=target_language.lower(),
                    update_session_after_seconds=40,
                    # sleep_seconds=0.5,
                )
                error = False
                i = 1
                return result
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    i += 1
                    sleep_seconds = 2**i
                    print(
                        f"Send too many requests, sleeping for {sleep_seconds} seconds"
                    )
                    time.sleep(sleep_seconds)
                else:
                    raise e
            except KeyError as e:
                return None
        raise ValueError("No response?")
    if translator == "deepl":
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": "3d89f756-7141-477c-bd19-76198a6aa447:fx",
            "text": text,
            "target_lang": target_language,
        }
        response = requests.post(url, data=params)
        result = response.json()
        if "translations" not in result:
            raise ValueError(result["message"])
        return result["translations"][0]["text"]

    raise ValueError(f"Unknown translator: {translator}")


def get_charset(
    target: str = "lang:ja",
    include_numerical: bool = True,
    include_punctuation: bool = True,
):
    charset_type, target = target.split(":")
    if charset_type == "script":
        charset = get_charset_by_script(target)
    else:
        charset = get_charset_by_language(
            target, include_numerical, include_punctuation
        )
    return charset


if __name__ == "__main__":
    for _ in range(100):
        label = generate_valid_unicode_label(
            get_charset_by_language("th"), 10
        )  # Generates a label of length 10 within the range U+0020 to U+007E
        # print(f"Generated label: {label}")
        # print(f"Generated alabel: {idna.encode(label).decode()}")
        print(idna.encode(label).decode())
