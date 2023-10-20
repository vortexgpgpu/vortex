#!/bin/sh

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
