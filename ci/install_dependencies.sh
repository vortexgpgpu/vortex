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

# Function to check if GCC version is less than 11
check_gcc_version() {
    local gcc_version
    gcc_version=$(gcc -dumpversion)
    if dpkg --compare-versions "$gcc_version" lt 11; then
        return 0  # GCC version is less than 11
    else
        return 1  # GCC version is 11 or greater
    fi
}

# Update package list
apt-get update -y

# install system dependencies
apt-get install -y build-essential valgrind libstdc++6 binutils python3 uuid-dev ccache cmake libffi7

# Check and install GCC 11 if necessary
if check_gcc_version; then
    echo "GCC version is less than 11. Installing GCC 11..."
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get update
    apt-get install -y g++-11 gcc-11
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
else
    echo "GCC version is 11 or greater. No need to install GCC 11."
fi
