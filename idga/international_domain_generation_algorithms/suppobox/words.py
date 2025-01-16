import requests
from faker import Faker
from tqdm import tqdm
import translators as ts

from idga import translate_text


def translate_file(fname, target_language: str = "JA", translator="deepl"):
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

    return translated_words
    # print(chars)
    # with open(f"{fname}_{target_language}", "w") as f:
    #     for word in translated_words:
    #         f.write(f"{word}\n")


def generate_words(
    word_set: int = 3, language: str = "th_TH", translator: str = "deepl"
):
    words = []

    if word_set == 1 or word_set == 2:
        words = translate_file(
            f"words{word_set}.txt", target_language=language, translator=translator
        )

    if word_set == 3:
        faker = Faker(language)
        for _ in range(384):
            words.append(faker.first_name())

    with open(f"words{word_set}_{language}.txt", "w") as file:
        for word in words:
            file.writelines(f"{word.split()[0]}\n")


if __name__ == "__main__":
    # languages = ["ES", "JA", "KR", "TR", "TH"]
    languages = ["TE"]
    for lang in languages:
        # generate_words(1, lang, translator="google")
        generate_words(2, lang, translator="bing")
        # generate_words(3, lang, translator="google")
