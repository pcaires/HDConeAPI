#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
make clean all
/bin/echo "DONE"