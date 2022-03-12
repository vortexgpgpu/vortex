
import sys

list_c = []
list_v = []

with open("golden_data/test_data.txt") as FH:
    lines = FH.readlines()
    for line in lines:
        line = line.strip()
        list_c.append(line)
with open("out_v.log") as FH:
    lines = FH.readlines()
    for line in lines:
        line = line.strip()
        list_v.append(line)

if len(list_v) != len(list_c):
    print("Matching failed")
    sys.exit(-1)

match_failed = False
list_v.sort()
for entry in list_v:
    if entry not in list_c:
        print(entry)
        match_failed = True

if match_failed:
    print("Matching failed")
    sys.exit(-1)
sys.exit(0)