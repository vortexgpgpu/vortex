#!/usr/bin/env python
import sys
import time
import threading
import subprocess

PingInterval = 15

def PingCallback(stop):
    wait_time = 0    
    while True:                   
        time.sleep(PingInterval)     
        wait_time += PingInterval   
        print(" + still running (" + str(wait_time) + "s) ...")
        sys.stdout.flush()        
        if stop(): 
            break

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print output.strip()
    return process.returncode

def main(argv):

    stop_threads = False
    t = threading.Thread(target = PingCallback, args =(lambda : stop_threads, ))
    t.start()

    exitcode = run_command(argv)
    print(" + exitcode="+str(exitcode))

    stop_threads = True
    t.join()

    sys.exit(exitcode)

if __name__ == "__main__":
   main(sys.argv[1:])