"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import json
import argparse
import os
from datetime import datetime
import hashlib

import idna
from idna import uts46_remap, IDNAError


class Rand:

    def __init__(self, seed, year, yday, offset=0):
        m = self.md5(seed)
        s = "{}{}{}".format(m, year, yday + offset)
        self.hashstring = self.md5(s)

    @staticmethod
    def md5(s):
        return hashlib.md5(s.encode("ascii")).hexdigest()

    def getval(self):
        v = int(self.hashstring[:8], 16)
        self.hashstring = self.md5(self.hashstring[7:])
        return v


def dga(date, maxdomainsfortry: int = 64, target_lang: str = "JA"):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    lang = target_lang.upper()
    path = os.path.join(current_directory, f"translated_words_{lang}.json")
    if not os.path.exists(path):
        print(f"Generate files for nymaim2-lang:{target_lang}")
        return []
    with open(path, "r") as r:
        wt = json.load(r)

    seed = "3138C81ED54AD5F8E905555A6623C9C9"
    daydelta = 10
    year = date.year % 100
    yday = date.timetuple().tm_yday - 1

    domains = []

    for dayoffset in range(daydelta + 1):
        r = Rand(seed, year, yday - dayoffset)
        error = False
        for _ in range(maxdomainsfortry):
            domain = ""
            for s in ["firstword", "separator", "secondword", "tld"]:
                ss = wt[s]
                if s == "tld":
                    label = uts46_remap(domain, std3_rules=False, transitional=False)
                    try:
                        idna.check_label(label)
                        domain = idna.encode(label).decode()
                    except IDNAError as e:
                        if e.args[0] == "Label too long":
                            error = True
                        elif e.args[0] == "Label must not start or end with a hyphen":
                            error = True
                        else:
                            raise e
                if error:
                    break
                domain += ss[r.getval() % len(ss)]
                domain = domain.replace(" ", "-")
            if error:
                error = False
                continue
            print(f"nymaim2-{target_lang}: {domain}")
            domains.append(domain)
            if len(domains) >= maxdomainsfortry:
                break
        if len(domains) >= maxdomainsfortry:
            break
    return domains


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    date = datetime.now()
    char_range, lang = target_charset.split(":")
    if char_range != "lang":
        raise ValueError(
            f"nymaim2 only compatible with language charset, not {char_range}"
        )
    domains = dga(date, amount, lang)
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
