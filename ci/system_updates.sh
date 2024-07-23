#!/bin/sh

# Copyright 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

apt-get update -y

add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y g++-11 gcc-11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100

apt-get install -y build-essential valgrind libstdc++6 binutils python uuid-dev ccache
