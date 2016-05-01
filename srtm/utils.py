# -*- coding: utf-8 -*-

# Copyright 2013 Tomo Krajina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import math
import logging    as mod_logging
import math       as mod_math
import zipfile    as mod_zipfile
try:
    import cStringIO as mod_cstringio
except:
    from io import StringIO as mod_cstringio
import matplotlib.pyplot as plt
import numpy as np

ONE_DEGREE = 1000. * 10000.8 / 90.

def distance(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    Distance between two points.
    """

    coef = mod_math.cos(latitude_1 / 180. * mod_math.pi)
    x = latitude_1 - latitude_2
    y = (longitude_1 - longitude_2) * coef

    return mod_math.sqrt(x * x + y * y) * ONE_DEGREE

def get_color_between(color1, color2, i):
    """ i is a number between 0 and 1, if 0 then color1, if 1 color2, ... """
    if i <= 0:
        return color1
    if i >= 1:
        return color2
    return (int(color1[0] + (color2[0] - color1[0]) * i),
            int(color1[1] + (color2[1] - color1[1]) * i),
            int(color1[2] + (color2[2] - color1[2]) * i))

def zip(contents, file_name):
    mod_logging.debug('Zipping %s bytes' % len(contents))
    result = mod_cstringio.StringIO()
    zip_file = mod_zipfile.ZipFile(result, 'w', mod_zipfile.ZIP_DEFLATED, False)
    zip_file.writestr(file_name, contents)
    zip_file.close()
    result.seek(0)
    mod_logging.debug('Zipped')
    return result.read()

def unzip(contents):
    mod_logging.debug('Unzipping %s bytes' % len(contents))
    zip_file = mod_zipfile.ZipFile(mod_cstringio.StringIO(contents))
    zip_info_list = zip_file.infolist()
    zip_info = zip_info_list[0]
    result = zip_file.open(zip_info).read()
    mod_logging.debug('Unzipped')
    return result

def get_path_length(latitude_1, longitude_1, latitude_2, longitude_2):
    '''calculates the distance between two lat, long coordinate pairs'''
    R = 6371000 # radius of earth in m
    latitude_1rads = math.radians(latitude_1)
    latitude_2rads = math.radians(latitude_2)
    deltaLat = math.radians((latitude_2-latitude_1))
    deltaLng = math.radians((longitude_2-longitude_1))
    a = math.sin(deltaLat/2) * math.sin(deltaLat/2) + math.cos(latitude_1rads) * math.cos(latitude_2rads) * math.sin(deltaLng/2) * math.sin(deltaLng/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

def get_destination_lat_long(latitude, longitude, azimuth, distance):
    '''returns the lat an long of destination point
    given the start lat, long, aziuth, and distance'''
    R = 6378.1 #Radius of the Earth in km
    brng = math.radians(azimuth) #Bearing is degrees converted to radians.
    d = distance/1000.0 #Distance m converted to km
    latitude_1 = math.radians(latitude) #Current dd lat point converted to radians
    longitude_1 = math.radians(longitude) #Current dd long point converted to radians
    latitude_2 = math.asin(math.sin(latitude_1) * math.cos(d/R) + math.cos(latitude_1)* math.sin(d/R)* math.cos(brng))
    lon2 = longitude_1 + math.atan2(math.sin(brng) * math.sin(d/R)* math.cos(latitude_1), math.cos(d/R)- math.sin(latitude_1)* math.sin(latitude_2))
    #convert back to degrees
    latitude_2 = math.degrees(latitude_2)
    lon2 = math.degrees(lon2)
    return[latitude_2, lon2]

def calculate_bearing(latitude_1, longitude_1, latitude_2, longitude_2):
    '''calculates the azimuth in degrees from start point to end point'''
    #print latitude_1, longitude_1, latitude_2, longitude_2
    startLat = math.radians(latitude_1)
    startLong = math.radians(longitude_1)
    endLat = math.radians(latitude_2)
    endLong = math.radians(longitude_2)
    dLong = endLong - startLong
    dPhi = math.log(math.tan(endLat/2.0+math.pi/4.0)/math.tan(startLat/2.0+math.pi/4.0))
    if abs(dLong) > math.pi:
         if dLong > 0.0:
             dLong = -(2.0 * math.pi - dLong)
         else:
             dLong = (2.0 * math.pi + dLong)
    bearing = (math.degrees(math.atan2(dLong, dPhi)) + 360.0) % 360.0;
    return bearing

def get_intermediate_coords(interval, azimuth, latitude_1, longitude_1,latitude_2,longitude_2):
    '''returns every coordinate pair inbetween two coordinate
    pairs given the desired interval'''

    d = get_path_length(latitude_1,longitude_1,latitude_2,longitude_2)
    remainder, dist = math.modf((d / interval))
    # print "from to ", latitude_1,longitude_1,latitude_2,longitude_2
    counter = float(interval)
    coords = []
    coords.append([latitude_1,longitude_1])
    for distance in xrange(0,int(dist)):
        coord = get_destination_lat_long(latitude_1,longitude_1,azimuth,counter)
        counter = counter + float(interval)
        coords.append(coord)
    coords.append([latitude_2,longitude_2])
    return coords

def metric_str(value):
    if value > 1000:
        return str(round(value/1000.0, 2)) + " km"
    else:
        return str(value) + " m"



def radius_elevation(elevation_data, latitude, longitude, distance, azimuth, angle):

    los_range = []

    for i in range(angle*2+1):
        n_azimuth = azimuth+i-angle
        if n_azimuth < 0:
            n_azimuth = 360 + n_azimuth
        if n_azimuth > 360:
            n_azimuth = n_azimuth -360

        dest_coord = get_destination_lat_long(latitude, longitude, n_azimuth, distance)
        coords = get_intermediate_coords(1, n_azimuth, latitude, longitude, dest_coord[0], dest_coord[1])
        # print elevation_data.get_elevation(dest_coord[0], dest_coord[1])
        elevation = []
        for coord in coords:
            elevation.append(elevation_data.get_elevation(coord[0], coord[1]))

        # elevation = [x+2 for x in elevation]

        visibility_range, dest_lat, dest_long = elevation_data.is_los(elevation, coords, distance)

        te = visibility_range if visibility_range != None else distance

        # dest_lat, dest_long = get_destination_lat_long(latitude, longitude, n_azimuth, te)
        los_range.append([n_azimuth, [dest_lat, dest_long]])
        print n_azimuth, te, dest_lat, dest_long
        print
    return los_range



    # plt.figure(figsize=(10,10))
    # ax = plt.gca()
    # x = [x/1000.0 for x in range(len(elevation))]
    # ax.fill_between(x, 0, elevation, color='#D3D3D3')

    # plt.plot(x, elevation, 'black')
    # plt.plot([0,len(elevation)/1000.0],[elevation[0], elevation[len(elevation)-1]], 'magenta')
    # ax.plot(0,elevation[0],marker='o',ms=10,mfc=(0,.9,0.,.78),mec='None')
    # ax.plot(len(elevation)/1000.0,elevation[len(elevation)-1],marker='o',ms=10,mfc=(.9,0.,0.,.78),mec='None')
    # plt.show()
