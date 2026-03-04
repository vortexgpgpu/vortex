#!/usr/bin/env bash
set -euo pipefail

echo "Installing Boost and OpenSSL from .deb packages"

WORKDIR=/var/tmp/vortex_pkgs
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# Boost
echo "Extracting Boost..."

mkdir -p "$WORKDIR/boost"
cp /boost-package.deb "$WORKDIR/boost/boost-package.deb"
cd "$WORKDIR/boost"

ar -x boost-package.deb

mkdir -p "$WORKDIR/boost-extract"
tar --zstd --no-same-owner -m -xvf data.tar.zst -C "$WORKDIR/boost-extract"

rm -rf /usr/local/boost-1.66
mkdir -p /usr/local
cp -a "$WORKDIR/boost-extract/opt/boost-1.66" /usr/local/

# OpenSSL
echo "Extracting OpenSSL..."

mkdir -p "$WORKDIR/openssl"
cp /openssl-package.deb "$WORKDIR/openssl/openssl-package.deb"
cd "$WORKDIR/openssl"

ar -x openssl-package.deb

mkdir -p "$WORKDIR/openssl-extract"
tar --zstd --no-same-owner -m -xvf data.tar.zst -C "$WORKDIR/openssl-extract"

rm -rf /usr/local/openssl-1.1
mkdir -p /usr/local
cp -a "$WORKDIR/openssl-extract/opt/openssl-1.1" /usr/local/

# Cleanup
cd /
rm -rf "$WORKDIR"

echo "Boost and OpenSSL installation complete!"