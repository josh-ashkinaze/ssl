"""
Date: 2025-05-24 13:44:34

Description: Gets the clean roles from ATUS data.

We accessed this page:
    https://www.atusdata.org/atus-action/variables/RELATEW#codes_section

At:
    2025-05-24 13:44:34

Input files:
- None
Output files:
- data/clean/atus_roles.txt: Cleaned roles from ATUS data
"""

#
# """
#
# Code	Label
# 03
# 04
# 05
# 06
# 07
# 08
# 09
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 0100	Alone	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# Household members
# 0200	Spouse	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0201	Unmarried partner	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0202	Own household child	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0203	Grandchild	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0204	Parent	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0205	Brother sister	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0206	Other related person	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0207	Foster child	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0208	Housemate, roommate	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0209	Roomer, boarder	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0210	Other nonrelative	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0300	Own non-household child under 18	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# Other non-household people
# 0400	Parents (not living in household)	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0401	Other non-household family members under 18	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0402	Other non-household family members 18 and older (including parents-in-law)	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0403	Friends	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0404	Co-workers, colleagues, clients (non-work activities only)	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0405	Neighbors, acquaintances	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0406	Other non-household children under 18	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0407	Other non-household adults 18 and older	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0408	Boss or manager (work activities only, 2010+)	·	·	·	·	·	·	·	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0409	People whom I supervise (work activities only, 2010+)	·	·	·	·	·	·	·	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# Code	Label
# 03
# 04
# 05
# 06
# 07
# 08
# 09
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 0410	Co-workers (work activities only, 2010+)	·	·	·	·	·	·	·	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 0411	Customers (work activities only, 2010+)	·	·	·	·	·	·	·	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 9996	Refused	X	·	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 9997	Don't know	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# 9998	Blank	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X	X
# """

import os
import logging
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)




d = [{"original": "Alone", "clean": "-1"},
{"original": "Spouse", "clean": "spouse"},
{"original": "Spouse", "clean": "husband"},
{"original": "Spouse", "clean": "wife"},
{"original": "Unmarried partner", "clean": "relationship partner"},
{"original": "Unmarried partner", "clean": "boyfriend"},
{"original": "Unmarried partner", "clean": "girlfriend"},
{"original": "Own household child", "clean": "child"},
{"original": "Grandchild", "clean": "grandchild"},
{"original": "Parent", "clean": "parent"},
{"original": "Parent", "clean": "mom"},
{"original": "Parent", "clean": "dad"},

{"original": "Brother sister", "clean": "sibling"},
{"original": "Brother sister", "clean": "brother"},
{"original": "Brother sister", "clean": "sister"},

{"original": "Other related person", "clean": "-1"},
{"original": "Foster child", "clean": "-1"},
{"original": "Housemate, roommate", "clean": "housemate"},
{"original": "Housemate, roommate", "clean": "roommate"},
{"original": "Roomer, boarder", "clean": "-1"},
{"original": "Other nonrelative", "clean": "-1"},
{"original": "Own non-household child under 18", "clean": "-1"},
{"original": "Parents (not living in household)", "clean": "-1"},
{"original": "Other non-household family members under 18", "clean": "-1"},
{"original": "Other non-household family members 18 and older (including parents-in-law)",
 "clean": "-1"},
{"original": "Friends", "clean": "friend"},
{"original": "Co-workers, colleagues, clients (non-work activities only)",
 "clean": "co-worker"},
{"original": "Co-workers, colleagues, clients (non-work activities only)",
 "clean": "colleague"}, #client overlaps with things like "email client'
{"original": "Neighbors, acquaintances", "clean": "neighbor"},
{"original": "Neighbors, acquaintances", "clean": "acquaintance"},
{"original": "Other non-household children under 18", "clean": "-1"},
{"original": "Other non-household adults 18 and older", "clean": "-1"},
{"original": "Boss or manager (work activities only, 2010+)", "clean": "boss"},
{"original": "Boss or manager (work activities only, 2010+)", "clean": "manager"},
{"original": "People whom I supervise (work activities only, 2010+)",
 "clean": "employee"},
{"original": "Co-workers (work activities only, 2010+)", "clean": "-1"},
{"original": "Customers (work activities only, 2010+)", "clean": "-1"}]


clean_roles = []
for i in d:
    clean = i["clean"]
    if clean != "-1":
        clean_roles.append(clean)

clean_roles = sorted(list(set(clean_roles)))

n_clean_roles = len(clean_roles)


print(f"Number of clean roles: {n_clean_roles}")
print(f"Clean roles: {clean_roles}")

logging.info(f"Number of clean roles: {n_clean_roles}")
logging.info(f"Clean roles: {clean_roles}")

with open("../data/clean/atus_roles.txt", "w") as f:
    for role in clean_roles:
        f.write(role + "\n")


# counselor, therapist, psychiatrist, teacher, coach, scout, advisor, counselor