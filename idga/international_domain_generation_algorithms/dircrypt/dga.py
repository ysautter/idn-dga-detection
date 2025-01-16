"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import argparse

import idna
from idna import uts46_remap

import idga


class RandInt:

    def __init__(self, seed):
        self.seed = seed

    def rand_int_modulus(self, modulus):
        ix = self.seed
        ix = 16807 * (ix % 127773) - 2836 * (ix // 127773) & 0xFFFFFFFF
        self.seed = ix
        return ix % modulus


def get_domains(seed, nr, target_charset="lang:ja"):
    r = RandInt(seed)
    # unicode_range = idga.get_ranges("Cyrillic")
    charset = idga.get_charset(
        target_charset, include_numerical=False, include_punctuation=False
    )
    for i in range(nr):
        length = 8
        error = True
        while error:
            try:
                domain_len = r.rand_int_modulus(12 + 1) + length
                domain = ""
                for i in range(domain_len):
                    # char = chr(ord("a") + r.rand_int_modulus(25 + 1))
                    random_index = r.rand_int_modulus(len(charset))
                    char = charset[random_index]
                    domain += char

                label = uts46_remap(domain, std3_rules=False, transitional=False)
                label = idga.remove_invalid_hypens(label)
                label = idga.remove_invalid_starting_modifier(label)
                idna.check_label(label)
                new_domain = idna.encode(label).decode()
                new_domain += ".com"
                error = False
                yield new_domain
                length = 8
            except idna.core.IDNAError as e:
                if isinstance(e, idna.core.IDNABidiError):
                    continue
                if str(e) == "Label too long":
                    length = 7
                    continue
                continue
                # raise e


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    # using the Tranco 1M list from 2023-07-13 as list of good domains.
    domains = []
    for domain in get_domains(int("1234", 16), amount, target_charset=target_charset):
        print(f"dircrypt-{target_charset}: {domain}")
        domains.append(domain)
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
