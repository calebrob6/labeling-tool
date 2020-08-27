#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time

import bottle
import argparse
import base64
import json

import numpy as np
import cv2

import rasterio

import uuid

import logging

from multiprocessing import Queue, Process
from queue import Empty

global SAMPLE_QUEUE, REPEAT_QUEUE, OUTPUT_QUEUE
SAMPLE_QUEUE = Queue()
REPEAT_QUEUE = Queue()
OUTPUT_QUEUE = Queue()

global MAX_QUEUE_SIZE, SAMPLE_SIZE
MAX_QUEUE_SIZE = 256
SAMPLE_SIZE = 240

global TABLE_SERVICE, INPUT_FN, OUTPUT_FN
TABLE_SERVICE = None
INPUT_FN = None
OUTPUT_FN = None

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

def do_options():
    '''This method is necessary for CORS to work (I think --Caleb)
    '''
    bottle.response.status = 204
    return

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def record_sample():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    # From https://stackoverflow.com/questions/31405812/how-to-get-client-ip-address-using-python-bottle-framework
    client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')

    num_times_labeled = data["num_times_labeled"]

    log_row = {
        "client_ip": client_ip,
        "out_time": str(data["time"]),
        "in_time": time.ctime(),
        "x": data["x"],
        "y": data["y"],
        "size": data["sample_size"],
        "user_label": data["label"],
        "number_of_times_labeled": num_times_labeled,
    }
    
    if num_times_labeled < 4:
        REPEAT_QUEUE.put((data["x"], data["y"], num_times_labeled+1))

    OUTPUT_QUEUE.put(log_row)

    bottle.response.status = 200
    return json.dumps(data)

def get_sample():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    (x, y, imagery, img_patch, num_times_labeled) = SAMPLE_QUEUE.get()

    r = 9
    midpoint = SAMPLE_SIZE // 2
    start_idx = midpoint-r-1
    size = r*2 + 2
    for i in range(start_idx, start_idx+size+1):
        rem = (i//3)%3
        color = [ 0 if rem==2 else 255, 255 if rem==0 else 0, 255 if rem==0 else 0 ]
        img_patch[start_idx,i] = color
        img_patch[i,start_idx] = color
        img_patch[start_idx+size,i] = color
        img_patch[i,start_idx+size] = color

    img1 = img_patch.copy()
    img2 = img_patch[60:-60,60:-60].copy()
    img3 = img_patch[start_idx+1:start_idx+size,start_idx+1:start_idx+size].copy()

    img1 = cv2.imencode(".jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))[1].tostring()
    img1 = base64.b64encode(img1).decode("utf-8")
    data["imgLarge"] = img1

    #print("Image 1 size: %d" % (len(img1)))

    img2 = cv2.imencode(".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))[1].tostring()
    img2 = base64.b64encode(img2).decode("utf-8")
    data["imgMedium"] = img2

    #print("Image 2 size: %d" % (len(img2)))

    t_h, t_w, t_c = img3.shape
    
    img3 = cv2.imencode(".png", cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))[1].tostring()
    img3 = base64.b64encode(img3).decode("utf-8")
    data["imgSmall"] = img3

    #print("Image 3 size: %d" % (len(img3)))

    img4 = imagery.copy()
    cv2.circle(img4, (x, y), 100, (255,0,0), thickness=-1)
    img4 = cv2.resize(img4, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) 
    img4 = cv2.imencode(".jpg", cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))[1].tostring()
    img4 = base64.b64encode(img4).decode("utf-8")
    data["imgHuge"] = img4

    #print("Image 4 size: %d" % (len(img4)))

    data["time"] = time.ctime()
    data["x"] = x
    data["y"] = y
    data["sample_size"] = SAMPLE_SIZE
    data["num_times_labeled"] = num_times_labeled

    bottle.response.status = 200
    return json.dumps(data)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def data_loader_process():

    nodata_val = 0.0
    with rasterio.open(INPUT_FN, "r") as f:
        imagery = np.rollaxis(f.read(), 0, 3)
        nodata_val = f.nodata
        assert imagery.shape[2] == 3, "3-band RGB imagery required"
    print("Finished pre-loading data")

    i = 0
    while True:
        if SAMPLE_QUEUE.qsize() < MAX_QUEUE_SIZE:
            print("[%s]\tSAMPLE_QUEUE: %d\tREPEAT_QUEUE: %d\tOUTPUT_QUEUE: %d" % (time.ctime(), SAMPLE_QUEUE.qsize(), REPEAT_QUEUE.qsize(), OUTPUT_QUEUE.qsize()))
            
            if not REPEAT_QUEUE.empty():
                x, y, num_times_labeled = REPEAT_QUEUE.get()
                img_patch = imagery[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE]
            else:
                num_times_labeled = 0
                
                any_nodata = True
                while any_nodata:
                    x = np.random.randint(0, imagery.shape[1]-SAMPLE_SIZE)
                    y = np.random.randint(0, imagery.shape[0]-SAMPLE_SIZE)
                    img_patch = imagery[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE]
                    any_nodata = np.any(np.sum(img_patch == nodata_val, axis=2) == 3)

            SAMPLE_QUEUE.put((x, y, imagery, img_patch, num_times_labeled))
            
            i += 1
        else:
            # sleep a bit, then check to see if the queue is full
            time.sleep(0.5)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def print_process():
    while True:
        if not OUTPUT_QUEUE.empty():
            row = OUTPUT_QUEUE.get()

            keys = sorted(row.keys())

            if not os.path.exists(OUTPUT_FN):
                with open(OUTPUT_FN, "w") as f:
                    f.write("%s\n" % (
                        ",".join(keys)
                    ))
                   
            with open(OUTPUT_FN, "a") as f:
                for i, key in enumerate(keys):
                    f.write(str(row[key]))
                    if i < len(keys)-1:
                        f.write(",")
                f.write("\n")
        else:
            time.sleep(0.5)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def root_app():
    return bottle.static_file("index.html", root="")

def favicon():
    return

def everything_else(filepath):
    return bottle.static_file(filepath, root="")

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    global TABLE_SERVICE, INPUT_FN, OUTPUT_FN
    parser = argparse.ArgumentParser(description="Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4042)
    
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Path to filename to crop samples from")
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Path to directory where output will be stored")
    parser.add_argument("--seed_data_fn", action="store", dest="seed_data_fn", type=str, help="Filename of seed data to put in the queue", default=None)

    args = parser.parse_args(sys.argv[1:])


    assert not os.path.exists(args.output_fn), "The output file already exists, data would be overwritten, exiting..."
    OUTPUT_FN = args.output_fn
    assert os.path.exists(args.input_fn), "The input file does not exist, exiting..."
    INPUT_FN = args.input_fn
    
    if args.seed_data_fn is not None:

        f = open(args.seed_data_fn, "r")
        lines = f.read().strip().split("\n")
        f.close()
        
        num_added = 0
        for line in lines:
            fn, x, y, num_times_labeled = line.split(",")
            x, y, num_times_labeled = int(x), int(y), int(num_times_labeled)
            REPEAT_QUEUE.put((fn, x, y, num_times_labeled))
            num_added += 1
        print("Pre-seeded the work queue with %d items" % (num_added))
    
    p1 = Process(target=data_loader_process)
    p1.start()

    p2 = Process(target=print_process)
    p2.start() 

    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)

    app.route("/recordSample", method="OPTIONS", callback=do_options)
    app.route('/recordSample', method="POST", callback=record_sample)

    app.route("/getSample", method="OPTIONS", callback=do_options)
    app.route('/getSample', method="POST", callback=get_sample)

    app.route('/', method="GET", callback=root_app)
    app.route('/favicon.ico', method="GET", callback=favicon)
    app.route('/<filepath:re:.*>', method="GET", callback=everything_else)

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False
    }
    app.run(**bottle_server_kwargs)

if __name__ == '__main__':
    main()
