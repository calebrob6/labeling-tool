import os
import pickle

import rasterio
import fiona
import shapely
import shapely.geometry

import rtree

NAIP_BLOB_ROOT = 'https://naipblobs.blob.core.windows.net/naip'

class NAIPTileIndex(object):
    TILES = None
    
    @staticmethod
    def lookup(geom):
        if NAIPTileIndex.TILES is None:
            assert all([os.path.exists(fn) for fn in [
                "data/tile_index/naip/tile_index.dat",
                "data/tile_index/naip/tile_index.idx",
                "data/tile_index/naip/tiles.p"
            ]]), "You do not have the correct files, did you setup the project correctly"
            NAIPTileIndex.TILES = pickle.load(open("data/tile_index/naip/tiles.p", "rb"))
        return NAIPTileIndex.lookup_naip_tile_by_geom(geom)

    @staticmethod
    def lookup_naip_tile_by_geom(geom):
        minx, miny, maxx, maxy = shapely.geometry.shape(geom).bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        geom = shapely.geometry.shape(geom)

        results = []
        tile_index = rtree.index.Index("data/tile_index/naip/tile_index")
        intersected_indices = list(tile_index.intersection(geom.bounds))
        for idx in intersected_indices:
            intersected_fn = NAIPTileIndex.TILES[idx][0]
            intersected_geom = NAIPTileIndex.TILES[idx][1]
            if intersected_geom.contains(geom):
                results.append(intersected_fn)
        tile_index.close()
        
        if len(results) > 0:
            return results
        else:
            if len(intersected_indices) > 0:
                raise ValueError("Error, there are overlaps with tile index, but no tile completely contains selection")
            else:
                raise ValueError("No tile intersections")


    @staticmethod
    def lookup_naip_tile_by_point(lat, lon):
        point = shapely.geometry.Point(lon, lat)

        results = []
        tile_index = rtree.index.Index("data/tile_index/naip/tile_index")
        intersected_indices = list(tile_index.intersection(point.bounds))
        for idx in intersected_indices:
            intersected_fn = NAIPTileIndex.TILES[idx][0]
            intersected_geom = NAIPTileIndex.TILES[idx][1]
            if intersected_geom.contains(point):
                results.append(intersected_fn)
        tile_index.close()

        if len(results) > 0:
            return results
        else:
            if len(intersected_indices) > 0:
                raise ValueError("Error, there are overlaps with tile index, but no tile completely contains selection")
            else:
                raise ValueError("No tile intersections")