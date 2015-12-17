#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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
import openravepy

from pymanoid_lite.trajectory import Trajectory
from hrp4 import HRP4
from numpy import array, dot, eye, cos, sin
from openravepy import matrixFromPose
from retiming import retime_whole_body_trajectory
from scipy.spatial import KDTree


FOOT_X = 0.120   # half-length
FOOT_Y = 0.065   # half-width
FOOT_Z = 0.027   # half-height

nb_segments = 5
TOPP_ndiscrsteps = 100


class Box(object):

    def __init__(self, env, name, dims, pose, color, transparency=0.):
        self.T = eye(4)
        self.env = env
        self.X = dims[0]
        self.Y = dims[1]
        self.Z = dims[2]
        self.name = name
        self.body = openravepy.RaveCreateKinBody(env, '')
        self.body.SetName(name)
        self.body.InitFromBoxes(array([
            array([0., 0., 0., self.X, self.Y, self.Z])]))
        self.set_color(color)
        env.Add(self.body)
        if pose is not None:
            self.set_transform_pose(pose)
        if transparency > 0:
            self.set_transparency(transparency)

    def set_color(self, color):
        r, g, b = \
            [.9, .5, .5] if color == 'r' else \
            [.2, 1., .2] if color == 'g' else \
            [.2, .2, 1.]  # if color == 'b'
        for link in self.body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetAmbientColor([r, g, b])
                geom.SetDiffuseColor([r, g, b])

    def set_visibility(self, visible):
        self.body.SetVisible(visible)

    def set_transparency(self, transparency):
        for link in self.body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def __del__(self):
        self.env.Remove(self.body)

    def set_transform_pose(self, pose):
        self.pose = pose
        self.T = matrixFromPose(pose)
        self.R = self.T[:3, :3]
        self.p = self.T[:3, 3]
        self.body.SetTransform(self.T)

    @property
    def corners(self):
        assert self.is_foot, "Trying to get corners of point contact"
        return [
            dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]]

    def collides_with(self, other_pseudo):
        return self.env.CheckCollision(self.body, other_pseudo.body)


class SupervoxelTree(object):

    def __init__(self, env, fname, transparency=0., dx=0.24, dz=1.38,
                 pitch=0.44):
        """

        dx -- X translation of the camera frame
        dz -- Z translation of the camera frame
        pitch -- pitch rotation of the camera frame

        """
        self.boxes = self.read_centers(fname, dx, dz, pitch)
        self.tree = KDTree([box.p[:2] for box in self.boxes])

    def read_centers(self, fname, dx, dz, pitch):
        boxes = []
        with open(fname, 'r') as f:
            for line in f:
                v = map(float, line.split(','))
                with env:
                    name = 'sv%d' % len(env.GetBodies())
                    x = -cos(pitch) * v[1] + sin(pitch) * v[2] + dx
                    y = v[0]
                    z = -sin(pitch) * v[1] - cos(pitch) * v[2] + dz
                    boxes.append(Box(env, name, dims=[0.05, 0.05, 0.05],
                                     pose=[1., 0., 0., 0., x, y, z],
                                     color='r'))
        return boxes

    def query(self, p, radius):
        indexes = self.tree.query_ball_point(p[0:2], radius)
        return [self.boxes[i] for i in indexes]


def init_openrave(fpath):
    env = openravepy.Environment()  # create openrave environment
    env.Load(fpath)
    env.SetViewer('qtcoin')
    viewer = env.GetViewer()
    cam_trans = array([
        [0.2936988, -0.51198474, 0.80722527, -1.43207502],
        [-0.95587863, -0.16268234, 0.24460276, -0.6278013],
        [0.00608842, -0.84344892, -0.53717488, 1.86361814],
        [0., 0., 0., 1.]])
    viewer.SetBkgndColor([.5, .7, .9])
    viewer.SetCamera(cam_trans)
    return env


def init_robot(env):
    hrp = HRP4(env, robot_name='HRP4')
    q_start = hrp.q_halfsit.copy()
    q_start[hrp.TRANS_X] -= 0.15
    hrp.set_dof_values(q_start)
    return hrp


def contacting_links_for_segment(segment_id):
    if segment_id == 1:
        return [hrp.left_foot, hrp.right_foot]
    elif segment_id == 2:
        return [hrp.left_foot]
    elif segment_id == 3:
        return [hrp.left_foot, hrp.right_foot]
    elif segment_id == 4:
        return [hrp.right_foot]
    elif segment_id == 5:
        return [hrp.left_foot, hrp.right_foot]
    raise Exception("bad segment ID")


def single_run(segment_id):
    segment = Trajectory.load('data/segment_%d.pos' % segment_id)
    contacting_links = contacting_links_for_segment(segment_id)

    print "\n\nWRENCH"
    from retiming import compute_GI_face
    compute_GI_face(contacting_links)

    print "\n\nFORCES"
    from retiming import compute_GI_face_forces
    compute_GI_face_forces(contacting_links)
    print "\n\n"

    hrp.play_trajectory(segment)
    retimed = retime_whole_body_trajectory(hrp, segment, contacting_links,
                                           TOPP_ndiscrsteps)
    hrp.play_trajectory(retimed)
    return segment, retimed


if __name__ == "__main__":
    assert len(sys.argv) in [1, 2], "check input format\n\n" \
        "Usage: %s [segment_id]\n" % sys.argv[0]

    env = init_openrave('env.xml')
    sv_tree = SupervoxelTree(env, 'data/sv_center.txt')
    hrp = init_robot(env)

    if len(sys.argv) == 2:
        segment_id = int(sys.argv[1])
        for _ in xrange(10):
            single_run(segment_id)
        exit()

    init_chunks, retimed_chunks = [], []
    for segment_id in range(1, nb_segments + 1):
        print "\n==============\nSegment_id =", segment_id
        segment, retimed = single_run(segment_id)
        init_chunks.append(segment)
        retimed_chunks.append(retimed)
    init_traj = Trajectory(init_chunks)
    retimed_traj = Trajectory(retimed_chunks)
    hrp.play_trajectory(retimed_traj)

    IPython.embed()
