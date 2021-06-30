import os
import sys
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--type", type=str,choices=['origin', 'grist',], help="run initial file or grist file")
parse.add_argument("--times", type=int, help="time to run code")
flags, unparsed = parse.parse_known_args(sys.argv[1:])

for i in range(flags.times):
    command = f"nohup python -u scripts/one_time_runner.py --type {flags.type} > {flags.type}_{flags.times}.log 2>&1 &"
    os.system(command)