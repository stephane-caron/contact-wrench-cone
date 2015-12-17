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


import cvxopt
import cvxopt.solvers
import pymanoid_sage
import numpy
import time
import vector

from numpy import array, dot, eye, hstack, vstack, zeros


cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

CONV_THRES = 1e-2
DEBUG = False
DOF_SCALE = 0.8  # additional scaling to avoid joint-limit saturation
GAIN = 1.
DOF_LIM_GAIN = 0.05  # NB: this can act as velocity limiter if not set properly
MAX_ITER = 150


def full_to_active(x, active_dofs):
    x_act = numpy.zeros(len(active_dofs))
    for i, dof in enumerate(active_dofs):
        x_act[i] = x[dof.index]
    return x_act


class IKError(Exception):

    def __init__(self, msg=None, q=None):
        self.msg = msg
        self.q = q

    def __str__(self):
        return self.msg


class SelfCollides(Exception):

    def __init__(self, hrp, q):
        self.msg = str(hrp.last_collision)
        self.q = q

    def __str__(self):
        return self.msg


class KinematicTask(object):

    def __init__(self, f, J):
        self.f = f
        self.J = J


class LinkFrameTask(KinematicTask):

    """
    Enforce a given pose for a frame attached to the link. The origin of the
    frame is taken as the point of coordinates local_origin in the link's
    reference frame (RF). The orientation is the same as that of the RF.
    """

    def __init__(self, robot, active_dofs, link, local_origin, target_pose):
        active_indexes = [dof.index for dof in active_dofs]
        index = link.GetIndex()

        def f():
            T = link.GetTransform()
            pose = link.GetTransformPose()
            pose[4:] += dot(T[0:3, 0:3], local_origin)
            return pose - target_pose

        def J():
            T = link.GetTransform()
            pose = link.GetTransformPose()
            pose[4:] += dot(T[0:3, 0:3], local_origin)
            rot, pos = pose[:4], pose[4:]
            J_trans = robot.rave.CalculateJacobian(index, pos)
            J_rot = robot.rave.CalculateRotationJacobian(index, rot)
            J_full = numpy.vstack([J_rot, J_trans])
            # NB: vstack has same order as GetTransformPose()
            return J_full[:, active_indexes]

        super(LinkFrameTask, self).__init__(f, J)


class COMTask(KinematicTask):

    """Enforce a given projection of the COM on a plane floor."""

    def __init__(self, robot, active_dofs, target_com):
        assert target_com.shape == (3,)
        active_indexes = [dof.index for dof in active_dofs]
        f = lambda: robot.compute_com() - target_com
        J = lambda: robot.compute_com_jacobian()[:, active_indexes]
        super(COMTask, self).__init__(f, J)
        self.target = target_com


class PrioritizedKinematics(object):

    def __init__(self, robot, active_dofs):
        self.active_dofs = active_dofs
        self.active_indexes = [dof.index for dof in self.active_dofs]
        self.nb_active_dof = len(active_dofs)
        self.robot = robot
        self.tasks = []

    def append_link_frame_task(self, link, local_origin, target_pose):
        new_task = LinkFrameTask(
            self.robot, self.active_dofs, link, local_origin, target_pose)
        return self.tasks.append(new_task)

    def append_com_task(self, target_com):
        new_task = COMTask(self.robot, self.active_dofs, target_com)
        return self.tasks.append(new_task)

    def show_debug_info(self, itnum):
        conv_vect = array([vector.norm(task.f()) for task in self.tasks])
        conv_str = ["%10.8f" % x for x in conv_vect]
        print "   %4d: %s" % (itnum, ' '.join(conv_str))
        for task in self.tasks:
            if type(task) is COMTask:
                com = self.robot.compute_com(self.robot.rave.GetDOFValues())
                pymanoid_sage.rave.display_box(
                    self.robot.env, com, box_id="COM", color='g',
                    thickness=0.01)
                pymanoid_sage.rave.display_box(
                    self.robot.env, task.target, box_id='Target', color='b',
                    thickness=0.01)

    def get_active_dof_limits(self):
        q_max = array([dof.ulim for dof in self.active_dofs])
        q_min = array([dof.llim for dof in self.active_dofs])
        q_avg = .5 * (q_max + q_min)
        q_dev = .5 * (q_max - q_min)
        q_max = q_avg + DOF_SCALE * q_dev
        q_min = q_avg - DOF_SCALE * q_dev
        return q_max, q_min

    @property
    def converged(self):
        conv_norms = (vector.norm(task.f()) for task in self.tasks)
        return max(conv_norms) < CONV_THRES

    def solve_in_place(self, q_start):
        self.robot.rave.SetDOFValues(q_start)
        self.robot.rave.SetActiveDOFs(self.active_indexes)
        q = full_to_active(q_start, self.active_dofs)
        q_max, q_min = self.get_active_dof_limits()
        I = eye(self.nb_active_dof)

        for itnum in xrange(MAX_ITER):
            if self.converged:
                break

            if DEBUG:
                self.show_debug_info(itnum)
                time.sleep(0.1)

            dq = zeros(self.nb_active_dof)
            dq_max = DOF_LIM_GAIN * (q_max - q)
            dq_min = DOF_LIM_GAIN * (q_min - q)
            Jstack, bstack = None, None
            for i, task in enumerate(self.tasks):
                Ji, bi = task.J(), -GAIN * task.f()

                # min.  || Ji * dq - bi ||
                # s.t.  dq_min <= dq <= dq_max
                # and   (Jj * dq) stays the same for all j < i
                qp_P = cvxopt.matrix(dot(Ji.T, Ji))
                qp_q = cvxopt.matrix(dot(-bi.T, Ji))
                qp_G = cvxopt.matrix(vstack([+I, -I]))
                qp_h = cvxopt.matrix(hstack([dq_max, -dq_min]))
                qp_args = [qp_P, qp_q, qp_G, qp_h]
                if Jstack is not None:
                    qp_A = cvxopt.matrix(Jstack)
                    qp_b = cvxopt.matrix(bstack)
                    qp_args.extend([qp_A, qp_b])
                qp_x = cvxopt.solvers.qp(*qp_args)['x']
                dq = array(qp_x).reshape((I.shape[0],))

                Js, bs = Ji, dot(Ji, dq)
                if type(task) is LinkFrameTask:
                    # removing one angular coord since cvxopt does not support
                    # redundancy in its equality constraints
                    Js, bs = Js[1:], bs[1:]
                Jstack = Js if Jstack is None else vstack([Jstack, Js])
                bstack = bs if bstack is None else hstack([bstack, bs])

            q += dq
            assert all(dq <= dq_max)
            assert all(dq_min <= dq)
            self.robot.rave.SetActiveDOFValues(q)
        return self.robot.rave.GetDOFValues()

    def solve(self, q_start):
        if DEBUG:
            q = self.solve_in_place(q_start)
        else:
            with self.robot.rave:
                q = self.solve_in_place(q_start)
        if self.robot.self_collides(q):
            raise SelfCollides(self.robot, q)
        elif not self.converged:
            raise IKError("did not converge", q)
        return q
