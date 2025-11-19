run:
    gcc -shared -fPIC -pthread -ldl -o mgm.so mgm.c

use ex:
    $ LD_PRELOAD=<path to repo>/minimal-gpu-monitor/mgm.so python3 benchmark.py 