import os
from subprocess import check_output
import re
from time import sleep

#
#  Feel free (a.k.a. you have to) to modify this to instrument your code
#

APPROACH = [1,2,3,4]
INPUTS = [
    "input/random-n2048-d16-c16.txt",
    "input/random-n16384-d24-c16.txt",
    "input/random-n65536-d32-c16.txt"
]
# INPUTS = ["input/input-n1024-d32-c16.txt",
# "input/input-n2048-d32-c16.txt",
# "input/input-n4096-d32-c16.txt",
# "input/input-n8192-d32-c16.txt",
# "input/input-n16384-d32-c16.txt",
# "input/input-n32768-d32-c16.txt",
# "input/input-n65536-d32-c16.txt",
# "input/input-n131072-d32-c16.txt",
# "input/input-n262144-d32-c16.txt",]
# INPUTS = ["input/input-n65536-d32-c16.txt"]

def extract_dim(inp):
    # Regular expression to match the number following 'd'
    match = re.search(r'-d(\d+)', inp)
    if match:
        return int(match.group(1))  # Extract and convert to integer
    return None

csvs = []
for inp in INPUTS:
    csv = [inp]
    for app in APPROACH:
        dim = extract_dim(inp)
        cmd = "./bin/kmeans -i {} -k 16 -d {} -m 150 -t 1e-6 -s 8675309 -c -a {}".format(
            inp, dim, app)
        out = check_output(cmd, shell=True).decode("ascii")
        m = re.search("Time per iteration: (.*)", out)
        if m is not None:
            time = m.group(1)
            csv.append(time)

    csvs.append(csv)
    sleep(0.5)

header = ["microseconds"] + [str(x) for x in APPROACH]


print("\n")
print(", ".join(header))
for row in csvs:
    print (", ".join(row))