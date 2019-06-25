#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

fswebcam -q -r 1280x720 --no-banner /opt/demo/images/$DATE.jpg
python3 -W ignore /opt/demo/test.py --image /opt/demo/images/$DATE.jpg 2>/dev/null
