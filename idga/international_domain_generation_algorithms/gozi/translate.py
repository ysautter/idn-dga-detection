import json

import requests
from tqdm import tqdm

from idga import translate_text


# def translate_text(text, target_language="DE"):
#     url = "https://api-free.deepl.com/v2/translate"
#     params = {
#         "auth_key": "3d89f756-7141-477c-bd19-76198a6aa447:fx",
#         "text": text,
#         "target_lang": target_language,
#     }
#     response = requests.post(url, data=params)
#     result = response.json()
#     return result["translations"][0]["text"]


def translate_file(fname, target_language: str = "JA", translator: str = "deepl"):
    with open(fname, "r") as f:
        words = [line.strip() for line in f]

    translated_words = []
    chars = 0
    for word in tqdm(words):
        chars += len(word)
        translated_word = translate_text(
            word, target_language=target_language, translator=translator
        )
        translated_words.append(translated_word)

    print(chars)
    with open(f"{fname}_{target_language}", "w") as f:
        for word in translated_words:
            f.write(f"{word}\n")


def main():
    lang = ["ML", "TE"]
    for l in lang:
        translate_file("gpl", l, translator="bing")
        # translate_file("luther", l, translator="google")
        # translate_file("nasa", l, translator="google")
        # translate_file("rfc4343", l, translator="google")


if __name__ == "__main__":
    main()
