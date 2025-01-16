"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
############
# CharBot implemetation from the paper in https://arxiv.org/abs/1905.01078
#
# When checking, I'm getting ~0% of registered domains, in agreement with the paper which says
# the algorithm generates 100% valid DGA domains.
#
import os

import dns.resolver
import idna
import pandas as pd
import numpy as np
import requests
from idna import uts46_remap

import idga

# from idga.idga import random_unicode_char, get_ranges
# from idga import idga

dnscharset = (
    [chr(x) for x in range(0x61, 0x61 + 26)]
    + [chr(x) for x in range(0x30, 0x30 + 10)]
    + ["-"]
)
TLDs = "com, at, uk, pl, be, biz, co, jp, cz, de, eu, fr, info, it, ru, lv, me, name, net, nz, org, us".split(
    ", "
)


def dns_query_specific_nameserver(
    query="techoverflow.net", nameserver="1.1.1.1", qtype="A"
):
    """
    Query a specific nameserver for:
    - An IPv4 address for a given hostname (qtype="A")
    - An IPv6 address for a given hostname (qtype="AAAA")

    Returns the IP address as a string
    """
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = [nameserver]
    answer = resolver.resolve(query, qtype)
    if len(answer) == 0:
        return None
    else:
        return str(answer[0])


def create_new_domain(tranco, target_charset="lang:ja", check_availability=False):
    real_domain = tranco.sample(1).domain.values[0]
    while len(real_domain.split(".")[0]) < 6:
        real_domain = tranco.sample(1).domain.values[0]
    new_domain = list(real_domain.split(".")[0])

    error = True
    while error:
        # index of the chars to replace in the read domain
        i, j = np.random.randint(0, len(new_domain), 2)
        # index of the chars to be replaced with
        # ci, cj = np.random.randint(0, len(dnscharset), 2)
        # index of the TLD
        tdlidx = np.random.randint(len(TLDs))

        # new_domain[i] = dnscharset[ci]
        # new_domain[j] = dnscharset[cj]
        charset = idga.get_charset(target_charset)
        new_domain[i] = idga.random_unicode_char(charset, i)
        new_domain[j] = idga.random_unicode_char(charset, j)
        try:
            label = uts46_remap(new_domain, std3_rules=False, transitional=False)
            idna.check_label(label)
            new_domain = idna.encode(label).decode()
            error = False
        except idna.core.IDNAError as e:
            if str(e) == "Label too long":
                real_domain = tranco.sample(1).domain.values[0]
                while len(real_domain.split(".")[0]) < 6:
                    real_domain = tranco.sample(1).domain.values[0]
                new_domain = list(real_domain.split(".")[0])
            continue

    new_domain += "." + TLDs[tdlidx]
    new_domain = "".join(new_domain)

    if check_availability:
        try:
            # if the DNS query returns an IP, then is not valid DGA
            dns_query_specific_nameserver(new_domain)
            valid = 0
            print("Not valid DGA: ", real_domain, new_domain)
        except (
            dns.resolver.NXDOMAIN,
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
        ):
            valid = 1
    else:
        valid = 1
    # returns the sampled domain, the transformed domain and 1 if is a valid DGA or 0 if new_domain is registered
    return [real_domain, new_domain, valid]


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    # using the Tranco 1M list from 2023-07-13 as list of good domains.
    np.random.seed(42)
    tranco = pd.read_csv("tranco1M.csv", names=["ranking", "domain"])

    # create 1000 valid domains
    domains = []
    print("Generating domains")
    for _ in range(amount):
        # Can take some time, as will query almost 1000 non existing domains
        dga = create_new_domain(
            tranco, check_availability=False, target_charset=target_charset
        )
        if dga[2] == 1:
            print(f"charbot-{target_charset}: {dga[1]}")
            domains.append(dga[1])
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
