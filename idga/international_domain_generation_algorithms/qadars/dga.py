"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import argparse
import time
from datetime import datetime, timedelta
import time
import string

import idna
from idna import uts46_remap

import idga


def rand(r, seed):
    return (seed - 1043968403 * r) & 0x7FFFFFFF


def dga(date, seed, amount: int = 200, target_charset: str = "lang:ja"):

    charset = idga.get_charset(target_charset)
    charset = [c for c in charset if c != "-"]
    # charset = string.ascii_lowercase + string.digits

    if seed in [0xE1F2, 0xE1F1, 0xE1F5]:
        tlds = [".com", ".org", ".net"]
    else:
        tlds = [".net", ".org", ".top"]
    unix = int(time.mktime(date.timetuple()))
    b = 7 * 24 * 3600
    c = 4 * 24 * 3600
    r = unix - (unix - c) % b
    domains = []
    for i in range(amount):
        domain = ""
        for _ in range(12):
            r = rand(r, seed)
            domain += charset[r % len(charset)]
        r = rand(r, seed)
        tld = tlds[r % 3]

        label = uts46_remap(domain, std3_rules=False, transitional=False)
        label = idga.remove_invalid_hypens(label)
        label = idga.remove_invalid_starting_modifier(label)
        if label == "":
            continue
        try:
            idna.check_label(label)
        except idna.IDNABidiError as e:
            continue
        domain = idna.encode(label).decode()

        domain += tld
        print(f"qadars-{target_charset}: {domain}")
        domains.append(domain)
    return domains


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    seeds = ["89f5", "4449", "E1F1", "E1F2", "E08A", "E1F5"]
    d = datetime.now()
    domains = set()
    while len(domains) <= amount:
        for seed in seeds:
            domains.update(
                dga(
                    d,
                    int(seed, 16),
                    amount=int(amount / len(seeds)),
                    target_charset=target_charset,
                )
            )
            if len(domains) >= amount:
                break
        d += timedelta(days=1)
        if len(domains) >= amount:
            break
    domains = list(domains)
    if len(domains) >= amount:
        domains = domains[:amount]
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
