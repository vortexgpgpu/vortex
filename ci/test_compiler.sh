#!/bin/bash

# Copyright © 2019-2023
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

# exit when any command fails
set -e

# clear POCL cache
rm -rf ~/.cache/pocl

# force rebuild test kernels
make -C tests clean-all

# ensure build
make -s

# run tests
make -C tests/kernel run-simx
make -C tests/regression run-simx
make -C tests/opencl run-simx