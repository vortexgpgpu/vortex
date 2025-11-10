#!/bin/bash -h
/bin/rm -rf `/bin/ls -A | /bin/grep -v "json" | /bin/grep -v "hsopt" | /bin/grep -v "product_timestamp" | /bin/grep -v "sysc"` >/dev/null 2>&1
