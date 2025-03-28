#!/bin/bash

set -e  # Exit on error
echo "Installing Boost and OpenSSL from .deb packages"

# Extract and install Boost
echo "Extracting Boost..."
ar -x boost-package.deb
ls
mkdir -p /tmp/boost-extract
tar --zstd --no-same-owner -xvf data.tar.zst -C /tmp/boost-extract
ls -l /tmp/boost-extract/
cp -r /tmp/boost-extract/opt/boost-1.66/ /opt/
ls -l /opt/boost-1.66/
rm -rf data.tar

# Extract and install OpenSSL
echo "Extracting OpenSSL..."
ar -x openssl-package.deb
mkdir -p /tmp/openssl-extract
tar --zstd --no-same-owner -xvf data.tar.zst -C /tmp/openssl-extract
ls -l /tmp/openssl-extract/
cp -r /tmp/openssl-extract/opt/openssl-1.1/ /opt/
ls -l /opt/openssl-1.1/
rm -rf data.tar



echo "Boost and OpenSSL installation complete!"
