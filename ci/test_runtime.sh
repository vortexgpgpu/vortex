#!/bin/bash

# exit when any command fails
set -e

make -C runtime/tests run
