"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import argparse
from datetime import datetime

import idna
from idna import uts46_remap

import idga


def init_rand_and_chars(year, month, day, nr_b, r, target_charset="lang:ja"):
    r = (r + year + ((nr_b << 16) + (month << 8) | day)) & 0xFFFFFFFF
    # charset = [chr(x) for x in range(ord("a"), ord("z"))] + [
    #     chr(x) for x in range(ord("0"), ord("9"))
    # ]
    # unicode_range = get_ranges("Greek")
    # charset = [chr(cp) for cp in unicode_range]
    # charset = idga.get_charset_by_language("el")

    charset = idga.get_charset(target_charset)
    charset = [c for c in charset if c != "-"]
    return charset, r


def generate_domain(charset, r):
    len_l = 0xC
    len_u = 0x18
    r = (1664525 * r + 1013904223) & 0xFFFFFFFF
    domain_len = len_l + r % (len_u - len_l)
    domain = ""
    new_domain = None
    for i in range(domain_len, 0, -1):
        r = ((1664525 * r) + 1013904223) & 0xFFFFFFFF
        domain += charset[r % len(charset)]
        if domain == "-":
            domain = ""
            continue
        label = uts46_remap(domain, std3_rules=False, transitional=False)
        label = idga.remove_invalid_hypens(label)
        label = idga.remove_invalid_starting_modifier(label)
        if label == "":
            continue
        try:
            idna.check_label(label)
        except idna.core.IDNABidiError as e:
            continue
        try:
            new_domain = idna.encode(label).decode()
        except idna.core.IDNAError as e:
            # Error is thrown, when idna is too long, we catch this, remove the last char and break
            domain = list(domain)
            del domain[-1]
            label = uts46_remap(domain, std3_rules=False, transitional=False)
            label = idga.remove_invalid_hypens(label)
            label = idga.remove_invalid_starting_modifier(label)
            idna.check_label(label)
            new_domain = idna.encode(label).decode()
            break
    if not new_domain:
        return generate_domain(charset, r)

    new_domain += ".ddns.net"
    return r, new_domain


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    d = datetime.now()
    day = d.day
    domains = []
    charset, r = init_rand_and_chars(
        d.year, d.month, day, 1, int("1DBA8930", 16), target_charset=target_charset
    )
    for _ in range(amount):
        r, domain = generate_domain(charset, r)
        print(f"corebot-{target_charset}: {domain}")
        domains.append(domain)
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
