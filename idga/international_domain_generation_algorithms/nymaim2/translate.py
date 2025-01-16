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


def main():
    with open("words.json", "r") as f:
        words = json.load(f)

    target_languages = ["ML", "TE"]

    for target_language in target_languages:

        translated = {
            "separator": ["-", ""],
            "firstword": [],
            "secondword": [],
            "tld": words["tld"],
        }
        # target_language = "JA"
        chars = 0
        keys = ["firstword", "secondword"]
        for key in keys:
            for word in tqdm(words[key]):
                chars += len(word)
                translated_word = translate_text(
                    word, target_language=target_language, translator="bing"
                )
                translated[key].append(translated_word)

        print(chars)
        with open(f"translated_words_{target_language}.json", "w") as f:
            json.dump(translated, f)


if __name__ == "__main__":
    main()
