#!/bin/sh
ps -A | grep xrcserver | awk '{print $1}' | xargs kill -9 $1
ps -A | grep loader | awk '{print $1}' | xargs kill -9 $1
ps -A | grep vpl | awk '{print $1}' | xargs kill -9 $1
ps -A | grep v++ | awk '{print $1}' | xargs kill -9 $1
ps -A | grep vivado | awk '{print $1}' | xargs kill -9 $1
ps -A | grep runme.sh | awk '{print $1}' | xargs kill -9 $1
ps -A | grep ISEWrap.sh | awk '{print $1}' | xargs kill -9 $1
ps -A | grep vrs | awk '{print $1}' | xargs kill -9 $1
ps -A | grep xcd | awk '{print $1}' | xargs kill -9 $1
ps -A | grep make | awk '{print $1}' | xargs kill -9 $1
