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

ps -A | grep debug_hw | awk '{print $1}' | xargs kill -9 $1
ps -A | grep python3 | awk '{print $1}' | xargs kill -9 $1
ps -A | grep xvc_pcie | awk '{print $1}' | xargs kill -9 $1
ps -A | grep hw_server | awk '{print $1}' | xargs kill -9 $1