"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import os
from datetime import datetime, timedelta
from ctypes import c_uint
import argparse

import idna
from idna import uts46_remap

seeds = {
    "luther": {"div": 4, "tld": ".com", "nr": 12},
    "rfc4343": {"div": 3, "tld": ".com", "nr": 10},
    "nasa": {"div": 5, "tld": ".com", "nr": 12},
    "gpl": {"div": 5, "tld": ".ru", "nr": 10},
}


class Rand:

    def __init__(self, seed):
        self.r = c_uint(seed)

    def rand(self):
        self.r.value = 1664525 * self.r.value + 1013904223
        return self.r.value


def get_words(wordlist, target_language: str = "JA"):
    target_language = target_language.upper()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_directory, f"{wordlist}_{target_language}")
    with open(path, "r") as r:
        return [w.strip() for w in r if w.strip()]


def dga(date, wordlist, amount: int = 100, target_lang: str = "ja"):
    words = get_words(wordlist, target_language=target_lang)
    diff = date - datetime.strptime("2015-01-01", "%Y-%m-%d")
    days_passed = diff.days // seeds[wordlist]["div"]
    flag = 1
    seed = (flag << 16) + days_passed - 306607824
    r = Rand(seed)

    for i in range(amount):
        r.rand()
        v = r.rand()
        length = v % 12 + 10
        domain = ""
        while len(domain) < length:
            v = r.rand() % len(words)
            word = words[v]
            l = len(word)
            if not r.rand() % 3:
                l >>= 1
            if len(domain) + l <= 24:
                domain += word[:l]

        label = uts46_remap(domain, std3_rules=False, transitional=False)
        try:
            idna.check_label(label)
            domain = idna.encode(label).decode()
            domain += seeds[wordlist]["tld"]

            yield domain
        except Exception as e:
            pass


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    # using the Tranco 1M list from 2023-07-13 as list of good domains.
    domains = set()
    d = datetime.now()
    char_range, lang = target_charset.split(":")
    if char_range != "lang":
        raise ValueError(
            f"Gozi only compatible with language charset, not {char_range}"
        )
    # wordlist: luther, nasa, rfc4343, gpl

    while len(domains) < amount:
        for domain in dga(d, "gpl", amount=amount, target_lang=lang):
            print(f"gozi-{target_charset}: {domain}")
            domains.add(domain)
            if len(domains) >= amount:
                break
        d = d + timedelta(days=1)
    return list(domains)


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
