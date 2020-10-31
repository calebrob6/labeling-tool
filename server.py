#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
import time
import copy
import argparse
import base64
import json
import uuid
import logging

from multiprocessing import Queue, Process, Manager
from queue import Empty

import numpy as np
import pandas as pd

import rasterio
import rasterio.mask
import rasterio.features
import fiona
import fiona.transform
import shapely
import shapely.geometry

import cv2
import bottle

import utils

global SAMPLE_QUEUE, OUTPUT_QUEUE, PENDING_LABELS, TIMEOUT_THRESHOLD
SAMPLE_QUEUE = Queue()
OUTPUT_QUEUE = Queue()
manager = Manager()
PENDING_LABELS = manager.dict()
TIMEOUT_THRESHOLD = 5 # in seconds

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

def reverse_coordinates_geom(geom):
    new_coords = []
    for x,y in geom["coordinates"][0]:
        new_coords.append((y,x))
    geom["coordinates"] = [new_coords]
    return geom

def filter_fns(fns):
    for fn in fns:    
        if "/ny/" in fn and "/2017/" in fn:
            return fn
        elif "/pa/" in fn and "/2017/" in fn:
            return fn
        elif "/de/" in fn and "/2018/" in fn:
            return fn
        elif "/md/" in fn and "/2018/" in fn:
            return fn
        elif "/va/" in fn and "/2018/" in fn:
            return fn
        elif "/wv/" in fn and "/2018/" in fn:
            return fn
    raise ValueError("No valid fn found: " + str(fns))

def get_imagery_from_geom(geom, src_crs):
    fns = utils.NAIPTileIndex.lookup_naip_tile_by_geom(geom)
    fn = filter_fns(fns)

    out_image_pairs = []

    with rasterio.open(utils.NAIP_BLOB_ROOT + "/" + fn) as f:
        dst_crs = f.crs.to_string()
        
        geom = reverse_coordinates_geom(copy.deepcopy(geom))
        geom_transformed = fiona.transform.transform_geom(src_crs, dst_crs, geom)
        
        shape_transformed = shapely.geometry.shape(geom_transformed)
        
        for buffer_amount in [50,100,200]:
        
            shape_transformed_buffered = shape_transformed.buffer(buffer_amount)
            geom_transformed_buffered = shapely.geometry.mapping(shape_transformed_buffered.envelope)

            out_image, out_transform = rasterio.mask.mask(f, [geom_transformed_buffered], crop=True, all_touched=True, filled=True)
            out_image = np.rollaxis(out_image, 0, 3)
            out_image = out_image[:,:,:3]

            out_mask = rasterio.features.geometry_mask([geom_transformed], (out_image.shape[0], out_image.shape[1]), out_transform)

            out_image_masked = out_image.copy()
            out_image_masked[~out_mask] = [255,0,0]

            out_image_pairs.append((out_image, out_image_masked))

    return out_image_pairs



def record_sample():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    # From https://stackoverflow.com/questions/31405812/how-to-get-client-ip-address-using-python-bottle-framework
    client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')

    sample_idx = data["sample_idx"]

    if sample_idx in PENDING_LABELS:

        log_row = {
            "client_ip": client_ip,
            "client_idx": data["client_idx"],
            "out_time": str(data["time"]),
            "in_time": time.ctime(),
            "sample_idx": data["sample_idx"],
            "user_label": data["user_label"]
        }
        
        OUTPUT_QUEUE.put(log_row)
        del PENDING_LABELS[sample_idx]

        bottle.response.status = 200
        return json.dumps(data)
    else:
        print("Received a label that we didn't actually ask for... ignoring")
        bottle.response.status = 200
        return json.dumps(data)

def get_sample():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    inputs = SAMPLE_QUEUE.get()
    geom = inputs["geom"]
    sample_idx = inputs["sample_idx"]
    src_crs = inputs["src_crs"]

    in_image_pairs = get_imagery_from_geom(geom, src_crs)
    
    out_image_pairs = []
    for img1, img2 in in_image_pairs:
        img1_str = cv2.imencode(".jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))[1].tobytes()
        img2_str = cv2.imencode(".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))[1].tobytes()

        img1_str = base64.b64encode(img1_str).decode("utf-8")
        img2_str = base64.b64encode(img2_str).decode("utf-8")

        out_image_pairs.append((img1_str, img2_str))

    data["time"] = time.ctime()
    data["sample_idx"] = sample_idx
    data["img_pairs"] = out_image_pairs

    PENDING_LABELS[sample_idx] = (time.time(), inputs)

    bottle.response.status = 200
    return json.dumps(data)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def data_loader_process(input_fn, existing_sample_idxs):
    existing_sample_idxs = set(existing_sample_idxs)

    geoms = []
    sample_idxs = []
    skipped_idxs = 0
    with fiona.open(input_fn) as f:
        src_crs = f.crs["init"]
        for row in f:
            sample_idx = row["properties"]["idx"]
            if row["properties"]["idx"] in existing_sample_idxs:
                skipped_idxs += 1
            else:
                geoms.append(row["geometry"])
                sample_idxs.append(sample_idx)

    print(f"Skipped {skipped_idxs} samples that have already been labeled")

    for i in range(len(geoms)):
        SAMPLE_QUEUE.put({
            "geom": geoms[i],
            "sample_idx": sample_idxs[i],
            "src_crs": src_crs
        })

    # Constantly monitor the PENDING_LABELS dictionary for items that have been sent out for labeling, but we haven't gotten an answer for in more than TIMEOUT_THRESHOLD seconds. If this happens, requeue that item.
    while True:
        try:
            del_list = []
            for sample_idx, (out_time, inputs) in PENDING_LABELS.items():
                if time.time() - out_time > TIMEOUT_THRESHOLD:
                    print("Time out for sample %d" % (sample_idx))
                    SAMPLE_QUEUE.put(inputs)
                    del_list.append(sample_idx)
            for sample_idx in del_list:
                del PENDING_LABELS[sample_idx]

        except Exception as e:
            print(e)
        time.sleep(2)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def results_process(output_fn):
    print(f"Saving results to {output_fn}")
    
    while True:
        if not OUTPUT_QUEUE.empty():
            row = OUTPUT_QUEUE.get()

            print(row)
            
            keys = sorted(row.keys())
            if not os.path.exists(output_fn):
                with open(output_fn, "w") as f:
                    f.write("%s\n" % (
                        ",".join(keys)
                    ))
                   
            with open(output_fn, "a") as f:
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
    parser = argparse.ArgumentParser(description="Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4042)
    
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Path to geojson of input polygons")
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Path to filename where output will be stored")

    args = parser.parse_args(sys.argv[1:])

    assert os.path.exists(args.input_fn), "The input file does not exist, exiting..."
    

    print("Loading NAIP index")
    try:
        utils.NAIPTileIndex.lookup({})
    except:
        pass
    print("Finishing loading NAIP index")


    existing_sample_idxs = []
    if os.path.exists(args.output_fn):
        df = pd.read_csv(args.output_fn)
        for sample_idx in df["sample_idx"].values:
            existing_sample_idxs.append(sample_idx)

    p1 = Process(target=data_loader_process, args=(args.input_fn, existing_sample_idxs,))
    p1.start()

    p2 = Process(target=results_process, args=(args.output_fn,))
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
