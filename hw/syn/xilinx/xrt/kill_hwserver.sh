#!/bin/bash

ps -A | grep debug_hw | awk '{print $1}' | xargs kill -9 $1
ps -A | grep python3 | awk '{print $1}' | xargs kill -9 $1
ps -A | grep xvc_pcie | awk '{print $1}' | xargs kill -9 $1
ps -A | grep hw_server | awk '{print $1}' | xargs kill -9 $1