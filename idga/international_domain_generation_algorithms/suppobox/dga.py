"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
"""
    generate domains according to: 
    - https://www.endgame.com/blog/malware-with-a-personal-touch.html
    - http://www.rsaconference.com/writable/presentations/file_upload/br-r01-end-to-end-analysis-of-a-domain-generating-algorithm-malware-family.pdf 

    requires words1.txt, words2.txt and words3.txt

    Thanks to Sándor Nemes who provided the third wordlist. It is taken
    from this sample:
    https://www.virustotal.com/en/file/4ee8484b95d924fe032feb8f26a44796f37fb45eca3593ab533a06785c6da8f8/analysis/
"""

import os
import time
from datetime import datetime
import argparse

import idna
from idna import uts46_remap


def generate_domains(time_, word_list, amount: int = 100, target_language: str = "JA"):
    domains = []
    target_language = target_language.upper()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        current_directory, "words{}_{}.txt".format(word_list, target_language)
    )
    with open(path, "r") as r:
        words = [w.strip() for w in r.readlines()]

    if not time_:
        time_ = time.time()
    seed = int(time_) >> 9
    for c in range(amount):
        nr = seed
        res = 16 * [0]
        shuffle = [3, 9, 13, 6, 2, 4, 11, 7, 14, 1, 10, 5, 8, 12, 0]
        for i in range(15):
            res[shuffle[i]] = nr % 2
            nr = nr >> 1

        first_word_index = 0
        for i in range(7):
            first_word_index <<= 1
            first_word_index ^= res[i]

        second_word_index = 0
        for i in range(7, 15):
            second_word_index <<= 1
            second_word_index ^= res[i]
        second_word_index += 0x80

        first_word = words[first_word_index]
        second_word = words[second_word_index]

        domain = "{}{}".format(first_word, second_word)
        label = uts46_remap(domain, std3_rules=False, transitional=False)
        label = label.replace(" ", "-")

        idna.check_label(label)
        domain = idna.encode(label).decode()

        tld = ".net"
        domain = "{}{}".format(domain, tld)
        domains.append(domain)
        print(f"suppobox-lang:{target_language.lower()}|{word_list}: {domain}")
        seed += 1
    return domains


def gen_iagds(amount=1000, target_charset: str = "lang:ja", word_list: int = 3):
    char_range, lang = target_charset.split(":")
    if char_range != "lang":
        raise ValueError(
            f"suppobox only compatible with language charset, not {char_range}"
        )
    datefmt = "%Y-%m-%d %H:%M:%S"
    date = datetime.now().strftime(datefmt)
    date = time.mktime(datetime.strptime(date, datefmt).timetuple())
    domains = generate_domains(date, word_list, amount=amount, target_language=lang)
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
