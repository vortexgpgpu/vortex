#!/usr/bin/env python

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

# This script executes a long-running command while outputing "still running ..." periodically
# to notify Travis build system that the program has not hanged

PING_INTERVAL=300 # 5 minutes

def monitor(stop):
    wait_time = 0    
    while True:                   
        time.sleep(PING_INTERVAL)     
        wait_time += PING_INTERVAL   
        print(" + still running (" + str(wait_time) + "s) ...")
        sys.stdout.flush()        
        if stop(): 
            break

def execute(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output:
            line = output.decode('utf-8').rstrip()
            print(">>> " + line)
            process.stdout.flush()
        ret = process.poll()
        if ret is not None:
            return ret        
    return -1

def main(argv):

    # start monitoring thread
    stop_monitor = False
    t = threading.Thread(target = monitor, args =(lambda : stop_monitor, ))
    t.start()

    # execute command
    exitcode = execute(argv)    
    print(" + exitcode="+str(exitcode))
    sys.stdout.flush()
    
    # terminate monitoring thread
    stop_monitor = True
    t.join()

    sys.exit(exitcode)

if __name__ == "__main__":
   main(sys.argv[1:])