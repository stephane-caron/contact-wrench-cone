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


import openravepy

from pymanoid_lite import Link, Manipulator, Robot
from numpy import array, pi, hstack


class HRP4(Robot):

    mass = 39.  # [kg]  (includes batteries)

    all_dofs = range(56)

    L_F53 = 0
    L_F52 = 1
    L_F43 = 2
    L_F42 = 3
    L_F33 = 4
    L_F32 = 5
    L_F23 = 6
    L_F22 = 7
    R_F53 = 8
    R_F52 = 9
    R_F43 = 10
    R_F42 = 11
    R_F33 = 12
    R_F32 = 13
    R_F23 = 14
    R_F22 = 15
    R_HIP_Y = 16
    R_HIP_R = 17
    R_HIP_P = 18
    R_KNEE_P = 19
    R_ANKLE_P = 20
    R_ANKLE_R = 21
    L_HIP_Y = 22
    L_HIP_R = 23
    L_HIP_P = 24
    L_KNEE_P = 25
    L_ANKLE_P = 26
    L_ANKLE_R = 27
    CHEST_P = 28
    CHEST_Y = 29
    NECK_Y = 30
    NECK_P = 31
    R_SHOULDER_P = 32
    R_SHOULDER_R = 33
    R_SHOULDER_Y = 34
    R_ELBOW_P = 35
    R_WRIST_Y = 36
    R_WRIST_P = 37
    R_WRIST_R = 38
    R_HAND_J0 = 39
    R_HAND_J1 = 40
    L_SHOULDER_P = 41
    L_SHOULDER_R = 42
    L_SHOULDER_Y = 43
    L_ELBOW_P = 44
    L_WRIST_Y = 45
    L_WRIST_P = 46
    L_WRIST_R = 47
    L_HAND_J0 = 48
    L_HAND_J1 = 49
    TRANS_X = 50
    TRANS_Y = 51
    TRANS_Z = 52
    ROT_R = 53
    ROT_P = 54
    ROT_Y = 55

    left_hand_dofs = [L_F53, L_F52, L_F43, L_F42, L_F33, L_F32, L_F23, L_F22,
                      L_HAND_J0, L_HAND_J1]

    right_hand_dofs = [R_F53, R_F52, R_F43, R_F42, R_F33, R_F32, R_F23, R_F22,
                       R_HAND_J0, R_HAND_J1]

    right_leg_dofs = [R_HIP_Y, R_HIP_R, R_HIP_P, R_KNEE_P, R_ANKLE_P, R_ANKLE_R]

    left_leg_dofs = [L_HIP_Y, L_HIP_R, L_HIP_P, L_KNEE_P, L_ANKLE_P, L_ANKLE_R]

    chest_dofs = [CHEST_P, CHEST_Y]

    neck_dofs = [NECK_Y, NECK_P]

    right_arm_dofs = [R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y, R_ELBOW_P,
                      R_WRIST_Y, R_WRIST_P, R_WRIST_R]

    left_arm_dofs = [L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y, L_ELBOW_P,
                     L_WRIST_Y, L_WRIST_P, L_WRIST_R]

    free_dofs = [TRANS_X, TRANS_Y, TRANS_Z, ROT_R, ROT_P, ROT_Y]

    q_halfsit = hstack([
        pi / 180 * array(
            # actuated DOF [deg]
            [0., 0., 0., 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,
                0., 0., -0.76, -22.02, 41.29, -18.75, -0.45, 0., 1.15, -21.89,
                41.21, -18.74, -1.10, 8., 0., 0., 0., -3., -10., 0., -30., 0.,
                0., 0., 0., 0., -3., 10., 0., -30., 0., 0., 0., 0.,  0.]),
        array(
            # unactuated in [m] and [rad]
            [0., 0., -0.0387, 0., 0., 0.])
    ])

    def __init__(self, env, robot_name='HRP4'):
        super(HRP4, self).__init__(env, robot_name)
        rave = self.rave
        self.register_collision_callback()
        self.left_foot = Manipulator(rave.GetManipulator("left_foot_center"))
        self.right_foot = Manipulator(rave.GetManipulator("right_foot_center"))
        self.left_hand = Manipulator(rave.GetManipulator("left_hand_palm"))
        self.right_hand = Manipulator(rave.GetManipulator("right_hand_palm"))
        self.left_arm = Link(rave.GetLink("L_ELBOW_P_LINK"))
        self.right_arm = Link(rave.GetLink("R_ELBOW_P_LINK"))

    def register_collision_callback(self):
        ignored_collision_pairs = [
            set(('CHEST_Y_LINK', 'R_SHOULDER_Y_LINK')),
            set(('CHEST_Y_LINK', 'L_SHOULDER_Y_LINK')),
            set(('R_HIP_Y_LINK', 'R_HIP_P_LINK')),
            set(('L_HIP_Y_LINK', 'L_HIP_P_LINK')),
            set(('BODY', 'R_SHOULDER_Y_LINK')),
            set(('BODY', 'L_SHOULDER_Y_LINK')),
        ]

        def callback(report, physics):
            name1 = report.plink1.GetName()
            name2 = report.plink2.GetName()
            if set((name1, name2)) in ignored_collision_pairs:
                print "ignored"
                return openravepy.CollisionAction.Ignore
            print "Collision:", report
            return openravepy.CollisionAction.DefaultAction

        # NB: once the handle loses scope, the callback is destroyed
        self.collision_handle = self.env.RegisterCollisionCallback(callback)
