#!/usr/bin/env python
import sys
import time
import threading
import subprocess

# This script executes a long-running command while outputing "still running ..." periodically
# to notify Travis build system that the program has not hanged

PING_INTERVAL=15

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
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output:
            line = output.decode('ascii').rstrip()
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