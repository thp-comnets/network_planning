#!/usr/bin/python
# -*- coding: utf-8 -*-

import srtm
import sys
from PIL import Image as mod_image
import ImageDraw as mod_imagedraw
import matplotlib.pyplot as plt
import srtm.utils as utils
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

import numpy as np
import math



# # Define points in great circle 1
# p1_lat1 = 32.498520
# p1_long1 = -106.816846
# p1_lat2 = 38.199999
# p1_long2 = -102.371389

# # Define points in great circle 2
# p2_lat1 = 34.086771
# p2_long1 = -107.313379
# p2_lat2 = 34.910553
# p2_long2 = -98.711786

# # Convert points in great circle 1, degrees to radians
# p1_lat1_rad = ((math.pi * p1_lat1) / 180.0)
# p1_long1_rad = ((math.pi * p1_long1) / 180.0)
# p1_lat2_rad = ((math.pi * p1_lat2) / 180.0)
# p1_long2_rad = ((math.pi * p1_long2) / 180.0)

# # Convert points in great circle 2, degrees to radians
# p2_lat1_rad = ((math.pi * p2_lat1) / 180.0)
# p2_long1_rad = ((math.pi * p2_long1) / 180.0)
# p2_lat2_rad = ((math.pi * p2_lat2) / 180.0)
# p2_long2_rad = ((math.pi * p2_long2) / 180.0)

# # Put in polar coordinates
# x1 = math.cos(p1_lat1_rad) * math.cos(p1_long1_rad)
# y1 = math.cos(p1_lat1_rad) * math.sin(p1_long1_rad)
# z1 = math.sin(p1_lat1_rad)
# x2 = math.cos(p1_lat2_rad) * math.cos(p1_long2_rad)
# y2 = math.cos(p1_lat2_rad) * math.sin(p1_long2_rad)
# z2 = math.sin(p1_lat2_rad)
# cx1 = math.cos(p2_lat1_rad) * math.cos(p2_long1_rad)
# cy1 = math.cos(p2_lat1_rad) * math.sin(p2_long1_rad)
# cz1 = math.sin(p2_lat1_rad)
# cx2 = math.cos(p2_lat2_rad) * math.cos(p2_long2_rad)
# cy2 = math.cos(p2_lat2_rad) * math.sin(p2_long2_rad)
# cz2 = math.sin(p2_lat2_rad)

# # Get normal to planes containing great circles
# # np.cross product of vector to each point from the origin
# N1 = np.cross([x1, y1, z1], [x2, y2, z2])
# N2 = np.cross([cx1, cy1, cz1], [cx2, cy2, cz2])

# # Find line of intersection between two planes
# L = np.cross(N1, N2)

# # Find two intersection points
# X1 = L / np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)
# X2 = -X1
# i_lat1 = math.asin(X1[2]) * 180./np.pi
# i_long1 = math.atan2(X1[1], X1[0]) * 180./np.pi
# i_lat2 = math.asin(X2[2]) * 180./np.pi
# i_long2 = math.atan2(X2[1], X2[0]) * 180./np.pi

# # Print results
# print i_lat1, i_long1, i_lat2, i_long2

map_size = 500.0

elevation_data = srtm.get_data(True, True, False, None, True)

# lat_from = 37.223
# lat_to = 37.223
# long_from = -122.092
# long_to = -122.407

# lat_from = 37.2478
# lat_to = 37.4432
# long_from = -122.27233
# long_to = -122.30941

# lat_from = 37.1478
# lat_to = 37.5432
# long_from = -122.27233
# long_to = -122.00941

# lat_from = 37.5021
# lat_to = 37.4814
# long_from = -122.395248
# long_to = -122.4337

# lat_from = 37.28061
# lat_to = 37.354876
# long_from = -122.166586
# long_to = -122.31353

#LOS
# lat_from = 37.751173
# lat_to = 37.67295
# long_from = -122.44262
# long_to = -122.133636

# 4 tiles US
# lat_from = 37.751173
# lat_to = 38.53312
# long_from = -122.44262
# long_to = -120.7109

# 4 tiles BRD
# lat_from = 50.6807
# lat_to = 51.134555
# long_from = 9.3164
# long_to = 11.6949

# manchester ->
lat_from = 38.980157
lat_to = 38.811491
long_from = -123.706832
long_to = -123.593231

# tersting
# lat_from = 38.985519
# lat_to = 38.985519
# long_from = -123.698753
# long_to = -123.657898

lat_from = 38.97642
lat_to = 38.967236
long_from = -123.7046
long_to = -123.699419

# real
lat_from = 38.970398
lat_to = 38.863
long_from = -123.700120
long_to = -123.645

lat_from = 37.809794
lat_to = 38.863
long_from = -122.716118
long_to = -123.645

# manchester one
lat_from = 38.980157
lat_to = 38.865587
long_from = -123.706832
long_to = -123.650862

# lat_from = 38.980157
# lat_to = 38.882277
# long_from = -123.706832
# long_to = -123.608264

print lat_from, long_from, lat_to,long_to
azimuth = utils.calculate_bearing(lat_from, long_from, lat_to, long_to)
print "Bearing:", round(azimuth,2), "°"
distance = utils.get_path_length(lat_from, long_from, lat_to, long_to)
print "Distance:", round(distance/1000 ,2), "km"
# don't change the interval!!
coords = utils.get_intermediate_coords(1, azimuth, lat_from, long_from, lat_to, long_to)

elevation = []
for coord in coords:
    elevation.append(elevation_data.get_elevation(coord[0], coord[1]))
peak, peak_coords = elevation_data.get_max_elevation_on_path(coords)
print "Highest peak on path:", peak , "m"

elevation_data.is_los(elevation, coords, distance)

print "##################################"
# res = elevation_data.find_path([lat_from, long_from], [lat_to, long_to], 163.189879874, 10000, 2222)
# if res:
#     print elevation_data.point_of_interest
#     print "********************************************"
#     print
#     print

# sys.exit()

az = 1
for i in range(16):

    print
    print
    print "---------", azimuth+az, az
    res = elevation_data.find_path([lat_from, long_from], [lat_to, long_to], azimuth+az, 10000, az)
    if res:
        print elevation_data.point_of_interest
        print "********************************************"
        print
        print
        # sys.exit()
    az = abs(az) +1 if az < 0 else  az * -1

gs = gridspec.GridSpec(2, 1,
                       width_ratios=[1],
                       height_ratios=[2,.5]
                       )


plt.figure(figsize=(10,10))

plt.subplot(gs[0])
image = elevation_data.get_image((map_size, map_size), sorted([lat_from, lat_to]), sorted([long_from, long_to]), 1000)
image = elevation_data.draw_line(image, lat_from, long_from, lat_to, long_to)
# coords = utils.radius_elevation(elevation_data, lat_from, long_from, 10000, azimuth, 45)
# image = elevation_data.draw_visibility(image, "red", coords)







# coords = utils.radius_elevation(elevation_data, lat_to, long_to, 10000, azimuth-180, 45)
# image = elevation_data.draw_visibility(image, "green", coords)
# image = elevation_data.draw_visibility(image, "red", lat_from, long_from, 10000, azimuth, 45)
# image = elevation_data.draw_visibility(image, "green", lat_to, long_to, 10000, azimuth-180, 45)
image.save("out.png")

img = mpimg.imread("out.png")
plt.imshow(img, aspect=1)

plt.tick_params(axis='both',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off',
                right='off',
                left='off',
                labelleft='off')

plt.subplot(gs[1])
ax = plt.gca()

elevation = [0 if v is None else v for v in elevation] # make sure None is not in the list
x = [x/1000.0 for x in range(len(elevation))]
ax.fill_between(x, 0, elevation, color='#D3D3D3')

plt.plot(x, elevation, 'black')
plt.plot([0,len(elevation)/1000.0],[elevation[0], elevation[len(elevation)-1]], 'magenta')
ax.plot(0,elevation[0],marker='o',ms=10,mfc=(.9,0,0.,.78),mec='None')
ax.plot(len(elevation)/1000.0,elevation[len(elevation)-1],marker='o',ms=10,mfc=(0.,0.9,0.,.78),mec='None')


plt.xlabel('Distance (km)')
plt.ylabel('Elevation (m)')
plt.grid(True)
plt.show()
sys.exit()



plt.tight_layout()
plt.savefig("map_elevation.pdf",
                    dpi=300,
                    bbox_inches='tight')
sys.exit()














image = elevation_data.get_image((map_size, map_size), (37, 38), (-123, -122), 1000)
lat_from = 37.223
long_from = -122.092
lat_to = 37.223
long_to = -122.407
print elevation_data.get_dist(lat_from, long_from, lat_to, long_to)

#image = elevation_data.draw_line(image, 37.223, -122.092, 37.223, -122.407)
image = elevation_data.draw_line(image, lat_from, long_from, lat_to, long_to)


elevation_data.get_elevations_on_line(lat_from, long_from, lat_to, long_to)

image.show()

#image.save("out.png")