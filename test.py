import time
import sys
import subprocess
import os
import base64
import uuid
import datetime
import traceback
import base64
import json
from time import gmtime, strftime
import math
import random, string
import time
import psutil
import uuid 
from getmac import get_mac_address
from coral.enviro.board import EnviroBoard
from luma.core.render import canvas
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from edgetpu.classification.engine import ClassificationEngine

# Importing socket library 
import socket 

start = time.time()

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

# Google Example Code
def update_display(display, msg):
    with canvas(display) as draw:
        draw.text((0, 0), msg, fill='white')

def getCPUtemperature():
    res = os.popen('vcgencmd measure_temp').readline()
    return(res.replace("temp=","").replace("'C\n",""))

# Get MAC address of a local interfaces
def psutil_iface(iface):
    # type: (str) -> Optional[str]
    import psutil
    nics = psutil.net_if_addrs()
    if iface in nics:
        nic = nics[iface]
        for i in nic:
            if i.family == psutil.AF_LINK:
                return i.address
# /opt/demo/examples-camera/all_models  
row = { }
try:
#i = 1
#while i == 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='File path of the image to be recognized.', required=True)
    args = parser.parse_args()
    # Prepare labels.
    labels = ReadLabelFile('/opt/demo/examples-camera/all_models/imagenet_labels.txt')

    # Initialize engine.
    engine = ClassificationEngine('/opt/demo/examples-camera/all_models/inception_v4_299_quant_edgetpu.tflite')

    # Run inference.
    img = Image.open(args.image)

    scores = {}
    kCount = 1

    # Iterate Inference Results
    for result in engine.ClassifyWithImage(img, top_k=5):
        scores['label_' + str(kCount)] = labels[result[0]]
        scores['score_' + str(kCount)] = "{:.2f}".format(result[1])
        kCount = kCount + 1    

    enviro = EnviroBoard()
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    cpuTemp=int(float(getCPUtemperature()))
    uuid2 = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
    end = time.time()
    row.update(scores)
    row['host'] = os.uname()[1]
    row['ip'] = host_ip
    row['macaddress'] = psutil_iface('eth0')
    row['cputemp'] = round(cpuTemp,2)
    row['end'] = '{0}'.format( str(end ))
    row['te'] = '{0}'.format(str(end-start))
    row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    row['cpu'] = psutil.cpu_percent(interval=1)
    usage = psutil.disk_usage("/")
    row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
    row['memory'] = psutil.virtual_memory().percent
    row['id'] = str(uuid2)
    row['message'] = "Success"
    row['temperature'] = '{0:.2f}'.format(enviro.temperature)
    row['humidity'] = '{0:.2f}'.format(enviro.humidity)
    row['tempf'] = '{0:.2f}'.format((enviro.temperature * 1.8) + 32)    
    row['ambient_light'] = '{0}'.format(enviro.ambient_light)
    row['pressure'] = '{0}'.format(enviro.pressure)
    msg = 'Temp: {0}'.format(row['temperature'])
    msg += 'IP: {0}'.format(row['ip'])
    update_display(enviro.display, msg)
#    i = 2
except:
    row['message'] = "Error"
print(json.dumps(row))
