#!/usr/bin/env python3

# Copyright 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
import threading
import subprocess

# This script executes a long-running command while printing "still running ..." periodically
# to notify Travis build system that the program has not hanged

PING_INTERVAL=300 # 5 minutes
SLEEP_INTERVAL=1  # 1 second

def monitor(stop_event):
    wait_time = 0
    elapsed_time = 0
    while not stop_event.is_set():
        time.sleep(SLEEP_INTERVAL)
        elapsed_time += SLEEP_INTERVAL
        if elapsed_time >= PING_INTERVAL:
            wait_time += elapsed_time
            print(" + still running (" + str(wait_time) + "s) ...")
            sys.stdout.flush()
            elapsed_time = 0

def execute(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output:
            try:
                line = output.decode('utf-8').rstrip()
            except UnicodeDecodeError:
                line = repr(output)  # Safely print raw binary data
            print(">>> " + line)
            process.stdout.flush()
        ret = process.poll()
        if ret is not None:
            return ret
    return -1

def main(argv):
    if not argv:
        print("Usage: travis_run.py <command>")
        sys.exit(1)

    # start monitoring thread
    stop_event = threading.Event()
    t = threading.Thread(target=monitor, args=(stop_event,))
    t.start()

    # execute command
    exitcode = execute(argv)
    print(" + exitcode="+str(exitcode))

    # terminate monitoring thread
    stop_event.set()
    t.join()

    sys.exit(exitcode)

if __name__ == "__main__":
   main(sys.argv[1:])