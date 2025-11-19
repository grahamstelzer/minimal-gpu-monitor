run:
    gcc -pthread -ldl -o mgm mgm.c

use:
    LD_PRELOAD=...<path to executable>.../minimal-gpu-monitor/mgm.so python3 vp-inference.py