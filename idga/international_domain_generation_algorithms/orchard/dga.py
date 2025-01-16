"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import argparse
import hashlib
import itertools
import json
import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Iterator, Union
import idna
import numpy as np
import homoglyphs as hg

import requests
from idna import uts46_remap

LIMIT = 100
DB_PATH = "db.json"
BLOCK = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
PATTERN = '{"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa":{"final_balance":FB,"n_tx":NTX,"total_received":FB}}'


def refresh_blockchain_db():
    offset = 0
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as r:
            db = json.load(r)
    else:
        db = {}

    while True:
        url = f"https://blockchain.info/multiaddr?active={BLOCK}&limit={LIMIT}&offset={offset}"
        r = requests.get(url)

        data = r.json()
        error = data.get('error')
        if error:
            print(f"error updating blockchain balance: {data}")
            quit()
        txs = data["txs"]
        for tx in txs:
            h = tx['hash']
            if h in db:
                break
            db[h] = tx
        else:
            time.sleep(1)
            offset += LIMIT
            continue
        break
    with open(DB_PATH, "w") as w:
        json.dump(db, w, indent=2)


def get_blockchain_seed(when, updated: bool = False) -> str:

    if when > datetime.now():
        raise ValueError(
            "you can't generate the blockchain domains for the future!")

    with open(DB_PATH) as r:
        transactions = json.load(r)

    transactions = sorted(
        transactions.values(),
        key=lambda x: x['balance'],
        reverse=True
    )
    ntx = len(transactions) + 1
    for i, transaction in enumerate(transactions):
        tt = transaction["time"]
        time = datetime.fromtimestamp(tt)

        if when < time:
            continue

        if i == 0 and not updated:
            """ if the desired date is later than the latest transaction,
                then update the transaction database to make sure it is
                the current transaction you like to access """
            refresh_blockchain_db()
            return get_blockchain_seed(when, updated=True)

        balance = transaction['balance']
        return PATTERN.replace("FB", str(balance)).replace("NTX", str(ntx-i-1))
    raise ValueError("the provided date is before the first transaction")


def dga(when: Union[date, datetime], blockchain: bool = False) -> Iterator[str]:
    dnscharset = (
            [chr(x) for x in range(0x61, 0x61 + 26)]
            + [chr(x) for x in range(0x30, 0x30 + 10)]
            + ["-"]
    )
    for i in range(2):
        if i and not blockchain:
            return

        if i == 0:
            magic = when.strftime("%Y-%m-%d")
            seed = f"{magic}ojena.duckdns.org"
        else:
            magic = get_blockchain_seed(when)
            seed = f"{magic}"

        md5 = hashlib.md5(seed.encode("ascii")).hexdigest()
        slds = [md5[i:i+8] for i in range(0, len(md5), 8)]
        tlds = [".com", ".net", ".org", ".duckdns.org"]

        for sld, tld in itertools.product(slds, tlds):
            error = True
            sld_orig = sld
            while error:
                try:
                    sld = list(sld_orig)
                    i, j = np.random.randint(0, len(sld), 2)
                    i_choices = set(hg.Homoglyphs().get_combinations(sld[i]))
                    i_choices = set([uts46_remap(x, std3_rules=False, transitional=False) for x in list(i_choices)])
                    i_choices -= set(dnscharset)
                    j_choices = set(hg.Homoglyphs().get_combinations(sld[j]))
                    j_choices = set([uts46_remap(x, std3_rules=False, transitional=False) for x in list(j_choices)])
                    j_choices -= set(dnscharset)
                    sld[i] = random.choice(list(i_choices))
                    sld[j] = random.choice(list(j_choices))
                    sld = "".join(sld)
                    label = uts46_remap(sld, std3_rules=False, transitional=False)
                    idna.check_label(label)
                    sld = idna.encode(label).decode()
                    error = False
                    yield f"{sld}{tld}"
                except Exception as e:
                    pass





def date_parser(s):
    return datetime.strptime(s, "%Y-%m-%d")

def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    domains = []
    today = datetime.now()
    days = [(today - timedelta(days=i)) for i in range(3 * 365)]

    for day in days:
        for domain in dga(day, False):
            print(f"orchard: {domain}")
            domains.append(domain)
            if len(domains) >= amount:
                break
        if len(domains) >= amount:
            break
    return domains

if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
