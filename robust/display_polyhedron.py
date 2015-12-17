#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Quang-Cuong Pham <cuong.pham@normalesup.org>
#
# This code is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this code. If not, see <http://www.gnu.org/licenses/>.


import sys
sys.path.append('../lib')

import IPython
import pymanoid_lite as pymanoid
import openravepy
import retiming
import time

from numpy import array, cross, dot, random, zeros
from hrp4 import HRP4

nb_iter = 1000000
plot_inside = True
plot_outside = False

env = openravepy.Environment()
env.Load('env.xml')

hrp = HRP4(env, robot_name='HRP4')
hrp.q_max[33] = -0.1  # R_SHOULDER_R (lower bound to avoid collision checks)
hrp.q_min[42] = +0.1  # L_SHOULDER_R (idem)

env.Remove(env.GetKinBody('foot_support'))
box = pymanoid.Robot(env, 'box')
box.body = box.rave.GetLink("BOX_BODY")
box.body.SetTransform(array([
    [0.99778209,  0.03872469,  0.05414154,  0.55237452],
    [-0.04629229,  0.98814841,  0.14635486, -0.10864417],
    [-0.04783233, -0.14853659,  0.98774944,  0.08722949],
    [0.,  0.,  0.,  1.]]))

arm_support = pymanoid.Robot(env, 'arm_support')
arm_support.body = arm_support.rave.GetLink("ARM_SUPPORT_BODY")
arm_support.set_dof_values([0.5, -0.4, 0.87, 0, 0, 0])
arm_support_origin = arm_support.body.GetTransformPose()[4:]

env.SetViewer('qtcoin')
viewer = env.GetViewer()
cam_trans = array([
    [0,  0, -1, 1.1],
    [1,  0,  0, 0.0],
    [0, -1,  0, 0.3],
    [0,  0,  0, 1.0]])
cam_trans[:, 3] *= 3  # step back for HRP4
viewer.SetBkgndColor([.5, .7, .9])
viewer.SetCamera(cam_trans)

hrp.collision_handle = None  # disable custom collision checker

hrp.set_dof_values(array(
    [-6.60631636e-01,  -6.60599394e-01,  -6.60631636e-01,
     -6.60599394e-01,  -6.60631636e-01,  -6.60599394e-01,
     -6.60631636e-01,  -6.60599394e-01,  -6.60638649e-01,
     -6.60680176e-01,  -6.60638649e-01,  -6.60680176e-01,
     -6.60638649e-01,  -6.60680176e-01,  -6.60638649e-01,
     -6.60680176e-01,   8.20765752e-02,  -4.61780744e-02,
     -5.63676339e-01,   1.25213970e+00,  -5.61702688e-01,
     -3.94210999e-01,  -3.21839045e-01,   1.92836363e-02,
     3.47070463e-01,   6.13913769e-01,   1.10621966e-01,
     -5.65984298e-01,  -1.89749246e-01,  -5.76178328e-01,
     -1.29370205e-03,   5.92799166e-03,  -7.33980307e-03,
     -1.53876550e-01,   2.68043028e-01,  -1.44070743e+00,
     -4.81258604e-03,  -3.79715153e-02,   4.27029982e-02,
     8.50404907e-01,   2.48255849e-03,  -3.01637955e-01,
     6.41684903e-01,   7.83369733e-02,  -5.62877258e-01,
     -1.07530159e-03,  -2.32347035e-03,   1.37502648e-02,
     -8.47950751e-01,   1.05909408e-03,   5.23164630e-01,
     5.89733869e-02 - 0.2,   7.02909529e-02 + 0.791 - 0.03,   2.48327606e-01,
     -1.01304501e-01,  -1.59442180e-01]))

robot = hrp.rave

t0 = time.time()
for i in range(100):
    CGI = retiming.compute_GI_face([hrp.right_foot, hrp.right_arm])
t1 = time.time()
print "Compute GI face (100 iter):", (t1-t0) * 1000 / 100, "ms / iter"

xm = 0.7
ym = -0.3
zm = 0.5
d = 1.5
points = []
g1 = array([0.15, 0, -0.98])
g2 = array([-0.15, 0, -0.98])
g3 = array([0, 0.15, -0.98])
g4 = array([0, -0.15, -0.98])
ft1, ft2, ft3, ft4 = zeros(6), zeros(6), zeros(6), zeros(6)
ft1[:3] = g1
ft2[:3] = g2
ft3[:3] = g3
ft4[:3] = g4

t0 = time.time()
for i in range(nb_iter):
    x = xm + d*(random.rand()-0.5)
    y = ym + d*(random.rand()-0.5)
    z = zm + 5*(random.rand()-0.5)
    pcom = array([x, y, z])
    ft1[3:] = cross(pcom, g1)
    ft2[3:] = cross(pcom, g2)
    ft3[3:] = cross(pcom, g3)
    ft4[3:] = cross(pcom, g4)
    if all(dot(CGI, ft1) <= 0) \
            and all(dot(CGI, ft2) <= 0) \
            and all(dot(CGI, ft3) <= 0) \
            and all(dot(CGI, ft4) <= 0):
        if plot_inside:
            points.append(env.plot3(pcom, 3, [0, 1, 0]))
    elif plot_outside:
        points.append(env.plot3(pcom, 5, [1, 0, 0]))
t1 = time.time()
print "Test (%d iter):" % nb_iter,
print (t1 - t0) * 1000 / 10000, "ms / iter"


cam_trans = array([
    [0,  0, -1, 3],
    [1,  0,  0, 0.0],
    [0, -1,  0, 0.7],
    [0,  0,  0, 1.0]])
viewer.SetCamera(cam_trans)
print "\nTo export:",
print "convert robust-x.png -crop 300x380+2250+800 +repage robust-x.pdf\n"
raw_input("Continue? ")

cam_trans = array([
    [1, 0, 0, 0.5],
    [0, 0, 1, -3],
    [0, -1, 0, 0.7],
    [0, 0, 0, 1]])
viewer.SetCamera(cam_trans)
print "\nTo export:",
print "convert robust-y.png -crop 300x380+2200+800 +repage robust-y.pdf\n"
raw_input("Continue? ")

cam_trans = array([
    [-1, 0, 0, 0.5],
    [0, 1, 0, -0.3],
    [0, 0, -1, 3],
    [0, 0, 0, 1]])
viewer.SetCamera(cam_trans)
print "\nTo export:",
print "convert robust-z.png -crop 300x380+2180+820 +repage robust-z.pdf\n"
raw_input("Continue? ")

IPython.embed()
