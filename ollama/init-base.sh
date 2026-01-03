#!/bin/bash
apt update
apt install -y --no-install-recommends curl
apt install -y --no-install-recommends ca-certificates
apt install -y --no-install-recommends jq
rm -rf /var/lib/apt/lists/*

