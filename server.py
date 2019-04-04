#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
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

import logging

from multiprocessing import Queue, Process

queue = Queue()
repeat_queue = Queue()

logpath = "output/labels.csv"
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
ch = logging.FileHandler(logpath)
#ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

TESTING=False
MAX_QUEUE_SIZE = 128
SAMPLE_SIZE = 240

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
    global queue, repeat_queue
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    # From https://stackoverflow.com/questions/31405812/how-to-get-client-ip-address-using-python-bottle-framework
    client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')

    user_labels = {
        "water": 1,
        "tree" : 2,
        "field": 3,
        "built": 4,
        "unknown": -1
    }
    label = user_labels[data["label"]]

    lc_labels = np.array(list(map(int, data["labels"].split(","))))
    size = int(np.sqrt(lc_labels.shape[0]))
    midpoint = (size-1)//2
    lc_labels = lc_labels.reshape(size, size)

    agrees = label == lc_labels[midpoint, midpoint]
    pct_agrees = np.sum(lc_labels == label) / (size**2)

    num_disagreements = data["num_disagreements"]

    log_row = [
        client_ip,
        time.ctime(),
        data["fn"],
        str(data["x"]),
        str(data["y"]),
        str(data["sample_size"]),
        str(label),
        str(num_disagreements),
        str(agrees),
        "%0.4f" % (pct_agrees),
        data["labels"]
    ]

    if not agrees and num_disagreements < 5:
        repeat_queue.put((data["fn"], data["x"], data["y"], num_disagreements+1))

    logger.info(','.join(log_row))

    bottle.response.status = 200
    return json.dumps(data)

def get_sample():
    global queue
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    fn, x, y, SAMPLE_SIZE, naip_data, naip_img, lc_img, num_disagreements = queue.get()

    r = 9
    midpoint = SAMPLE_SIZE // 2
    start_idx = midpoint-r-1
    size = r*2 + 2
    for i in range(start_idx, start_idx+size+1):
        naip_img[start_idx,i] = [255,0,0]
        naip_img[i,start_idx] = [255,0,0]
        naip_img[start_idx+size,i] = [255,0,0]
        naip_img[i,start_idx+size] = [255,0,0]

    img1 = naip_img.copy()
    img2 = naip_img[60:-60,60:-60].copy()
    img3 = naip_img[start_idx:start_idx+size+1,start_idx:start_idx+size+1].copy()

    img1 = cv2.imencode(".jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))[1].tostring()
    img1 = base64.b64encode(img1).decode("utf-8")
    data["imgLarge"] = img1

    #print("Image 1 size: %d" % (len(img1)))

    img2 = cv2.imencode(".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))[1].tostring()
    img2 = base64.b64encode(img2).decode("utf-8")
    data["imgMedium"] = img2

    #print("Image 2 size: %d" % (len(img2)))

    t_h, t_w, t_c = img3.shape
    cv2.circle(img3, (t_h//2, t_w//2), 4, (0,0,0), 1)
    img3 = cv2.imencode(".png", cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))[1].tostring()
    img3 = base64.b64encode(img3).decode("utf-8")
    data["imgSmall"] = img3

    #print("Image 3 size: %d" % (len(img3)))

    img4 = naip_data.copy()
    cv2.circle(img4, (x, y), 100, (255,0,0), thickness=-1)
    img4 = cv2.resize(img4, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) 
    img4 = cv2.imencode(".jpg", cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))[1].tostring()
    img4 = base64.b64encode(img4).decode("utf-8")
    data["imgHuge"] = img4

    #print("Image 4 size: %d" % (len(img4)))

    data["fn"] = fn
    data["x"] = x
    data["y"] = y
    data["sample_size"] = SAMPLE_SIZE
    data["labels"] = ",".join(lc_img[midpoint-r:midpoint+r+1, midpoint-r:midpoint+r+1].flatten().astype(str))
    data["num_disagreements"] = num_disagreements

    bottle.response.status = 200
    return json.dumps(data)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def queue_loader(queue, repeat_queue):

    if TESTING:
        f = open("data/training_sets_testing.txt","r")
    else:
        f = open("data/training_sets.txt","r")
    fns = f.read().strip().split("\n")
    f.close()

    naip_tiles = []
    lc_tiles = []
    for naip_fn in fns:
        print("Loading %s" % (naip_fn))
        f = rasterio.open(naip_fn, "r")
        naip_data = f.read()
        naip_data = np.rollaxis(naip_data, 0, 3)
        f.close()

        lc_fn = naip_fn.replace("esri-naip", "resampled-lc")[:-4] + "_lc.tif"
        f = rasterio.open(lc_fn, "r")
        lc_data = f.read().squeeze()
        f.close()

        lc_data[lc_data == 5] = 4
        lc_data[lc_data == 6] = 4
        lc_data[lc_data > 6] = 0

        naip_tiles.append(naip_data)
        lc_tiles.append(lc_data)
    print("Finished pre-loading data")

    i = 0
    while True:
        if queue.qsize() < MAX_QUEUE_SIZE:
            
            num_disagreements = 0

            if not repeat_queue.empty():
                print("Re-adding disagreed sample into queue")
                fn, x, y, num_disagreements = repeat_queue.get()
                idx = fns.index(fn)

                naip_data = naip_tiles[idx]
                lc_data = lc_tiles[idx]
            else:

                idx = np.random.randint(0, len(fns))
                fn = fns[idx]
                naip_data = naip_tiles[idx]
                lc_data = lc_tiles[idx]

                x = np.random.randint(0, naip_data.shape[1]-SAMPLE_SIZE)
                y = np.random.randint(0, naip_data.shape[0]-SAMPLE_SIZE)

            naip_img = naip_data[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE, :3]
            lc_img = lc_data[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE]

            queue.put((fn, x, y, SAMPLE_SIZE, naip_data[:,:,:3], naip_img, lc_img, num_disagreements))
            
            i += 1
        else:
            # sleep a bit, then check to see if the queue is full
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
    global queue, repeat_queue
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4042)

    args = parser.parse_args(sys.argv[1:])

    p = Process(target=queue_loader, args=(queue,repeat_queue))
    p.start()

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
