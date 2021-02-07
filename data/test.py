import os
import json

from pathlib import Path

import csv

with open("TLGAN.csv", 'r') as f:
    csvfile = csv.reader(f, delimiter=',')
    for row in csvfile:
        print(row)