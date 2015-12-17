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


from ik import PrioritizedKinematics


class Stance(object):

    def __init__(self, hrp, lfoot_pose, rfoot_pose, q_start=None, com=None):
        assert lfoot_pose.shape == (7,)
        assert rfoot_pose.shape == (7,)
        assert com is None or com.shape == (3,)

        self.com = com
        self.hrp = hrp
        self.q = q_start if q_start is not None else hrp.q_halfsit
        self.lfoot_link = hrp.left_foot_link
        self.lfoot_origin = hrp.left_foot_local_origin
        self.lfoot_pose = lfoot_pose.copy()
        self.rfoot_link = hrp.right_foot_link
        self.rfoot_origin = hrp.right_foot_local_origin
        self.rfoot_pose = rfoot_pose.copy()

        # recompute q and COM from IK
        self.update()

    def update(self, lfoot_pose=None, rfoot_pose=None, com=None):
        if lfoot_pose is not None:
            self.lfoot_pose = lfoot_pose
        if rfoot_pose is not None:
            self.rfoot_pose = rfoot_pose
        if com is not None:
            self.com = com
        self.q = self.compute_q(self.q)
        self.com = self.hrp.compute_com(self.q)

    def add_to_com(self, dcom):
        self.update(com=self.com + dcom)

    def add_to_lfoot_pose(self, d_lfoot_pose):
        self.update(lfoot_pose=self.lfoot_pose + d_lfoot_pose)

    def add_to_rfoot_pose(self, d_rfoot_pose):
        self.update(rfoot_pose=self.rfoot_pose + d_rfoot_pose)

    def copy(self):
        return Stance(
            self.hrp, self.lfoot_pose, self.rfoot_pose, self.q, self.com)

    def compute_q(self, q_start):
        act_dof_names = ['R_LEG', 'CHEST', 'R_ARM', 'L_ARM', 'L_LEG',
                         'TRANS_Y', 'TRANS_X', 'TRANS_Z']
        act_dofs = self.hrp.get_dofs(*act_dof_names)
        ik = PrioritizedKinematics(self.hrp, act_dofs)
        ik.append_link_frame_task(
            self.lfoot_link, self.lfoot_origin, self.lfoot_pose)
        ik.append_link_frame_task(
            self.rfoot_link, self.rfoot_origin, self.rfoot_pose)
        if self.com is not None:
            ik.append_com_task(self.com)
        return ik.solve(q_start)
