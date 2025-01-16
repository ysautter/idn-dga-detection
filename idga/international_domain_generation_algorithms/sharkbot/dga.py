"""Source code copied from domain_generation_algorithms (https://github.com/baderj/domain_generation_algorithms)
 and adapted to output IDNs."""
import argparse
import base64
import hashlib
import random
from datetime import datetime, timedelta

import idna
import numpy as np
from idna import uts46_remap

import homoglyphs as hg

dnscharset = (
    [chr(x) for x in range(0x61, 0x61 + 26)]
    + [chr(x) for x in range(0x30, 0x30 + 10)]
    + ["-"]
)


def dga(date: datetime, version: str):

    # this will actually be locale dependent, so be prepared for + or -
    # on week shifts
    week_of_year = date.isocalendar()[1]
    year = date.year
    if version in {"0.0", "1.63"}:
        tlds = [".top", ".xyz", ".cc", ".info", ".com", ".ru", ".net"]
    else:
        tlds = [".xyz", ".live", ".com", ".store", ".info", ".top", ".net"]

    for tld in tlds:
        if version == "2.8":
            s = f"{tld}{week_of_year}{year}"
            h = hashlib.md5(s.encode("ascii")).hexdigest()
            sld = h[:16]
            error = True
            sld_orig = sld
            while error:
                try:
                    sld = list(sld_orig)
                    i, j = np.random.randint(0, len(sld), 2)
                    i_choices = set(hg.Homoglyphs().get_combinations(sld[i]))
                    i_choices = set(
                        [
                            uts46_remap(x, std3_rules=False, transitional=False)
                            for x in list(i_choices)
                        ]
                    )
                    i_choices -= set(dnscharset)
                    j_choices = set(hg.Homoglyphs().get_combinations(sld[j]))
                    j_choices = set(
                        [
                            uts46_remap(x, std3_rules=False, transitional=False)
                            for x in list(j_choices)
                        ]
                    )
                    j_choices -= set(dnscharset)
                    sld[i] = random.choice(list(i_choices))
                    sld[j] = random.choice(list(j_choices))
                    sld = "".join(sld)
                    label = uts46_remap(sld, std3_rules=False, transitional=False)
                    idna.check_label(label)
                    sld = idna.encode(label).decode()
                    error = False
                except Exception as e:
                    pass
        elif version == "2.1":
            s = f"{tld}{week_of_year}"
            h = hashlib.md5(s.encode("ascii")).hexdigest()
            t = f"{h}{year}"
            sld = t[:16]
        elif version == "1.63":
            s = f"{week_of_year + year}pojBI9LHGFdfgegjjsJ99hvVGHVOjhksdf"
            b = base64.b64encode(s.encode("ascii"))
            sld = b[:19].decode("ascii").lower()
        elif version == "0.0":
            s = f"{week_of_year}pojBI9LHGFdfgegjjsJ99hvVGHVOjhksdf"
            b = base64.b64encode(s.encode("ascii"))
            sld = b[:19].decode("ascii").lower()
        else:
            raise ValueError(f"invalid version {version}")
        yield f"{sld}{tld}"


def gen_iagds(amount=1000, target_charset: str = "lang:ja"):
    domains = []
    date = datetime.now()

    days = [(date - timedelta(days=i)) for i in range(3 * 365)]

    for day in days:
        for domain in dga(date=day, version="2.8"):
            print(f"sharkbot: {domain}")
            domains.append(domain)
            if len(domains) >= amount:
                break
        if len(domains) >= amount:
            break
    return domains


if __name__ == "__main__":
    iagds = gen_iagds()
    print(len(iagds))
