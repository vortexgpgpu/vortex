#!/bin/sh

ps -A | grep launch_hw_emu.s | awk '{print $1}' | xargs kill -9 $1
ps -A | grep simulate.sh | awk '{print $1}' | xargs kill -9 $1
ps -A | grep xsim | awk '{print $1}' | xargs kill -9 $1
ps -A | grep xsimk | awk '{print $1}' | xargs kill -9 $1