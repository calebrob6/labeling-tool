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

from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

import logging

from multiprocessing import Queue, Process
from queue import Empty

DATASET_NAME = "denver_1"

global SAMPLE_QUEUE, REPEAT_QUEUE, OUTPUT_QUEUE
SAMPLE_QUEUE = Queue()
REPEAT_QUEUE = Queue()
OUTPUT_QUEUE = Queue()

global MAX_QUEUE_SIZE, SAMPLE_SIZE
MAX_QUEUE_SIZE = 256
SAMPLE_SIZE = 240

global TABLE_SERVICE, OUTPUT_PATH
TABLE_SERVICE = None
OUTPUT_PATH = None

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

    user_labels = {
        "structures": 1,
        "impervious" : 2,
        "water": 3,
        "grassland": 4,
        "tree": 5,
        "turf": 6,
        "barren": 7,
        "unknown": -1
    }

    label = user_labels[data["label"]]

    lc_labels = np.array(list(map(int, data["labels"].split(","))))
    size = int(np.sqrt(lc_labels.shape[0]))
    midpoint = (size-1)//2
    lc_labels = lc_labels.reshape(size, size)

    agrees = label == lc_labels[midpoint, midpoint]
    pct_agrees = np.sum(lc_labels == label) / (size**2)

    num_times_labeled = data["num_times_labeled"]

    log_row = [
        client_ip,
        str(data["time"]),
        time.ctime(),
        data["fn"],
        str(data["x"]),
        str(data["y"]),
        str(data["sample_size"]),
        str(label),
        str(num_times_labeled),
        str(agrees),
        "%0.4f" % (pct_agrees),
        data["labels"]
    ]

    if num_times_labeled < 4:
        REPEAT_QUEUE.put((data["fn"], data["x"], data["y"], num_times_labeled+1))

    OUTPUT_QUEUE.put(log_row)

    bottle.response.status = 200
    return json.dumps(data)

def get_sample():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    fn, x, y, sample_size, naip_data, naip_img, lc_img, num_times_labeled = SAMPLE_QUEUE.get()

    lc_img[lc_img == 8] = 4

    r = 9
    midpoint = sample_size // 2
    start_idx = midpoint-r-1
    size = r*2 + 2
    for i in range(start_idx, start_idx+size+1):
        rem = (i//3)%3
        color = [ 0 if rem==2 else 255, 255 if rem==0 else 0, 255 if rem==0 else 0 ]
        naip_img[start_idx,i] = color
        naip_img[i,start_idx] = color
        naip_img[start_idx+size,i] = color
        naip_img[i,start_idx+size] = color

    img1 = naip_img.copy()
    img2 = naip_img[60:-60,60:-60].copy()
    img3 = naip_img[start_idx+1:start_idx+size,start_idx+1:start_idx+size].copy()

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

    img4 = naip_data.copy()
    cv2.circle(img4, (x+120, y+120), 20, (255,0,0), thickness=-1)
    #img4 = cv2.resize(img4, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) 
    img4 = cv2.imencode(".jpg", cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))[1].tostring()
    img4 = base64.b64encode(img4).decode("utf-8")
    data["imgHuge"] = img4

    #print("Image 4 size: %d" % (len(img4)))

    data["time"] = time.ctime()
    data["fn"] = fn
    data["x"] = x
    data["y"] = y
    data["sample_size"] = sample_size
    data["labels"] = ",".join(lc_img[midpoint-r:midpoint+r+1, midpoint-r:midpoint+r+1].flatten().astype(str))
    data["num_times_labeled"] = num_times_labeled

    bottle.response.status = 200
    return json.dumps(data)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def data_loader_process():

    f = open("data/denver_fns.csv", "r")
    naip_fns = []
    lc_fns = []
    lines = f.read().strip().split("\n")
    for line in lines:
        t_fns = line.split(",")
        naip_fns.append(t_fns[0])
        lc_fns.append(t_fns[1])
    f.close()

    naip_tiles = []
    lc_tiles = []
    for i in range(len(naip_fns)):
        print("Loading %d/%d" % (i+1, len(naip_fns)))

        f = rasterio.open(naip_fns[i], "r")
        naip_data = f.read()
        naip_data = np.rollaxis(naip_data, 0, 3)
        f.close()

        f = rasterio.open(lc_fns[i], "r")
        lc_data = f.read()
        lc_data = lc_data.squeeze()
        f.close()

        naip_tiles.append(naip_data)
        lc_tiles.append(lc_data)
    print("Finished pre-loading data")

    i = 0
    while True:
        if SAMPLE_QUEUE.qsize() < MAX_QUEUE_SIZE:
            
            num_times_labeled = 0

            print("[%s]\tSAMPLE_QUEUE: %d\tREPEAT_QUEUE: %d\tOUTPUT_QUEUE: %d" % (time.ctime(), SAMPLE_QUEUE.qsize(), REPEAT_QUEUE.qsize(), OUTPUT_QUEUE.qsize()))

            if not REPEAT_QUEUE.empty():
                fn, x, y, num_times_labeled = REPEAT_QUEUE.get()
                idx = naip_fns.index(fn)
                
                naip_data = naip_tiles[idx]
                lc_data = lc_tiles[idx]
            else:
                idx = np.random.randint(0, len(naip_fns))
                fn = naip_fns[idx]
                naip_data = naip_tiles[idx]
                lc_data = lc_tiles[idx]

                x = np.random.randint(0, naip_data.shape[1]-SAMPLE_SIZE)
                y = np.random.randint(0, naip_data.shape[0]-SAMPLE_SIZE)

            naip_img = naip_data[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE, :3]
            lc_img = lc_data[y:y+SAMPLE_SIZE, x:x+SAMPLE_SIZE]

            if np.sum(np.sum(naip_img == 0, axis=2) == 3) / float(SAMPLE_SIZE*SAMPLE_SIZE) > 0.01:
                continue

            SAMPLE_QUEUE.put((fn, x, y, SAMPLE_SIZE, naip_data[:,:,:3], naip_img, lc_img, num_times_labeled))
            
            i += 1
        else:
            # sleep a bit, then check to see if the queue is full
            time.sleep(0.5)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def print_process():
    
    using_azure_tables = TABLE_SERVICE is not None

    while True:
        if not OUTPUT_QUEUE.empty():
            row = OUTPUT_QUEUE.get()

            if using_azure_tables:
                log_row = {
                    "PartitionKey": str(np.random.randint(0, 16)),
                    "RowKey": str(uuid.uuid4()),
                    "client_ip": row[0],
                    "out_time": row[1],
                    "in_time": row[2],
                    "file_name": row[3],
                    "x": row[4],
                    "y": row[5],
                    "size": row[6],
                    "user_label": row[7],
                    "number_of_times_labeled": row[8],
                    "agrees_with_ground_truth": row[9],
                    "percent_agreement_of_19m_neighborhood": row[10],
                    "neighborhood_19m": row[11],
                    "dataset": DATASET_NAME
                }
                TABLE_SERVICE.insert_entity("randomlabeltool", log_row)
            else:
                f = open(OUTPUT_PATH, "a")
                f.write("%s\n" % (",".join(row)))
                f.close()
        else:
            time.sleep(0.5)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def root_app():
    return bottle.static_file("index_denver.html", root="")

def favicon():
    return

def everything_else(filepath):
    return bottle.static_file(filepath, root="")

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    global TABLE_SERVICE, OUTPUT_PATH
    parser = argparse.ArgumentParser(description="Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4042)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--azure_table', action="store_true", dest="azure_table")
    group.add_argument("--output_path", action="store", dest="output_path", type=str, help="Path to directory where output will be stored")

    parser.add_argument("--seed_data_fn", action="store", dest="seed_data_fn", type=str, help="Filename of seed data to put in the queue", default=None)

    args = parser.parse_args(sys.argv[1:])

    if args.azure_table:
        assert "AZURE_ACCOUNT_NAME" in os.environ
        assert "AZURE_ACCOUNT_KEY" in os.environ
        print("Setting up TABLE_SERVICE")
        TABLE_SERVICE = TableService(
            account_name=os.environ['AZURE_ACCOUNT_NAME'],
            account_key=os.environ['AZURE_ACCOUNT_KEY']
        )
    else:
        assert not os.path.exists(args.output_path), "The output file already exists, data would be overwritten"
        print("Setting up OUTPUT_PATH")
        OUTPUT_PATH = args.output_path

    
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
