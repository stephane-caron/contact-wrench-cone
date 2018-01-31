#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
#
# pymanoid is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

import openravepy
import time

from numpy import arange, array, cross, dot, eye
from numpy import zeros, ones, hstack, vstack, tensordot


# Notations:
#
# c: link COM
# m: link mass
# omega: link angular velocity
# r: origin of link frame
# R: link rotation
# T: link transform
# v: link velocity (v = [rd, omega])
#
# unless otherwise mentioned, coordinates are in the absolute reference frame.


def crossmat(x):
    return array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])


def middot(M, T):
    """Dot product of a matrix with the mid-coordinate of a 3D tensor.

    M -- matrix with shape (n, m)
    T -- tensor with shape (a, m, b)

    Outputs a tensor of shape (a, n, b).

    """
    return tensordot(M, T, axes=(1, 1)).transpose([1, 0, 2])


def update_kinbody(env, name, aabb, color='r'):
    acolor = array([.1, .1, .1])
    dcolor = array([.1, .1, .1])
    cdim = 0 if color == 'r' else 1 if color == 'g' else 2
    acolor[cdim] += .2
    dcolor[cdim] += .4
    prec = env.GetKinBody(name)
    if prec is not None:
        env.Remove(prec)
    area = openravepy.RaveCreateKinBody(env, '')
    area.SetName(name)
    area.InitFromBoxes(array([array(aabb)]), True)
    g = area.GetLinks()[0].GetGeometries()[0]
    g.SetAmbientColor(acolor)
    g.SetDiffuseColor(dcolor)
    env.Add(area, True)


def display_box(env, pos, box_id='Box', thickness=0.01, color='r'):
    x, y, z = pos
    aabb = [x, y, z, thickness, thickness, thickness]
    update_kinbody(env, box_id, aabb, color)


def display_force(env, pos, vec, box_id='Force', f_scale=1e-3, thickness=5e-3):
    x, y, z = pos + .5 * f_scale * vec
    dx, dy, dz = .5 * f_scale * abs(vec)
    thickify = lambda x: x if abs(x) > thickness else x * thickness / abs(x)
    dx, dy, dz = map(thickify, [dx, dy, dz])
    aabb = [x, y, z, dx, dy, dz]
    update_kinbody(env, box_id, aabb)


class RaveRobotModel(object):

    chains = {}  # list of kinematic chains

    def __init__(self, env, robot_name):
        robot = env.GetRobot(robot_name)
        robot.GetEnv().GetPhysicsEngine().SetGravity(array([0, 0, -9.81]))
        dof_llim, dof_ulim = robot.GetDOFLimits()
        n = robot.GetDOF()

        vel_lim = robot.GetDOFVelocityLimits()
        tau_lim = 100000 * ones(n)
        for dof in self.dofs:
            # internal limits override those of the robot model
            if dof.vel_limit is not None:
                vel_lim[dof.index] = dof.vel_limit
            if dof.torque_limit is not None:
                tau_lim[dof.index] = dof.torque_limit
        robot.SetDOFVelocityLimits(1000 * vel_lim)  # current OpenRAVE bug

        self.dof_llim = dof_llim
        self.dof_ulim = dof_ulim
        self.env = env
        self.mass = sum([lnk.GetMass() for lnk in robot.GetLinks()])
        self.nb_active_dof = 0
        self.nb_dof = n
        self.rave = robot
        self.torque_limits = tau_lim

        for chain, joints in self.chains.iteritems():
            for joint in joints:
                for dof in joint.dofs:
                    dof.llim = self.dof_llim[dof.index]
                    dof.ulim = self.dof_ulim[dof.index]
                joint.ulim = self.dof_ulim[joint.dof_range]
                joint.llim = self.dof_llim[joint.dof_range]

    def get_dof_values(self):
        return self.rave.GetDOFValues()

    def set_dof_values(self, q):
        self.rave.SetDOFValues(q)

    @property
    def dofs(self):
        for chain, joints in self.chains.iteritems():
            for joint in joints:
                for dof in joint.dofs:
                    yield dof

    def get_dofs(self, *args):
        dofs = [dof
                for chain, joints in self.chains.iteritems()
                for joint in joints
                for dof in joint.dofs
                for identifier in [chain, joint.name, dof.name]
                if identifier in args]
        l = list(set(dofs))
        # convention: DOF ordered by index
        l.sort(key=lambda dof: dof.index)
        return l

    def get_dof(self, dof_name):
        return self.get_dofs(dof_name)[0]

    def play_trajectory(self, traj, callback=None, dt=3e-2, start=0.,
                        stop=None, nowait=False, slowdown=1.):
        if stop is None:
            stop = traj.duration
        trange = list(arange(start, stop, dt))
        if stop - trange[-1] >= dt:
            trange.append(stop)
        for t in trange:
            q = traj.q(t)
            qd = traj.qd(t)
            qdd = traj.qdd(t)
            self.rave.SetDOFValues(q)
            if callback:
                callback(t, q, qd, qdd)
            if not nowait:
                time.sleep(slowdown * dt)

    def start_recording(self, fname='output.mpg', codec=13, framerate=24,
                        width=800, height=600):
        vname = self.env.GetViewer().GetName()
        cmd = 'Start %d %d %d codec %d timing simtime filename %s\nviewer %s'
        cmd = cmd % (width, height, framerate, codec, fname, vname)
        recorder = openravepy.RaveCreateModule(self.env, 'viewerrecorder')
        self.env.AddModule(recorder, '')
        recorder.SendCommand(cmd)
        self.recorder = recorder

    def stop_recording(self):
        self.recorder.SendCommand('Stop')
        self.env.Remove(self.recorder)
        self.recorder = None

    def record_trajectory(self, traj):
        self.rave.SetDOFValues(traj.q(0))
        self.start_recording()
        time.sleep(1.)
        self.play_trajectory(traj)
        time.sleep(1.)
        self.stop_recording()

    def self_collides(self, q):
        assert len(q) in [self.nb_dof, self.nb_active_dof]
        with self.rave:  # need to lock environment when calling robot methods
            if len(q) == self.nb_dof:
                self.rave.SetDOFValues(q)
            else:  # len(q) == self.nb_active_dof
                self.set_active_dof_values(q)
            return self.rave.CheckSelfCollision()

    def compute_link_pose(self, link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            return link.GetTransformPose()

    def compute_link_jacobian(self, link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            index = link.GetIndex()
            pose = link.GetTransformPose()
            J_trans = self.rave.ComputeJacobianTranslation(index, pose[4:])
            J_rot = self.rave.ComputeJacobianAxisAngle(index)
            return vstack([J_rot, J_trans])

    def compute_link_hessian(self, link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            index = link.GetIndex()
            pose = link.GetTransformPose()
            H_trans = self.rave.ComputeHessianTranslation(index, pose[4:])
            H_rot = self.rave.ComputeHessianAxisAngle(index)
            return hstack([H_rot, H_trans])

    def compute_link_pose_jacobian(self, link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            index = link.GetIndex()
            pose = link.GetTransformPose()
            rot, pos = pose[:4], pose[4:]
            J_trans = self.rave.CalculateJacobian(index, pos)
            J_quat = self.rave.CalculateRotationJacobian(index, rot)
            return vstack([J_quat, J_trans])

    def compute_com(self, q):
        total = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                c = link.GetGlobalCOM()
                total += m * c
        return total / self.mass

    def compute_com_velocity(self, q, qd):
        total = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                R = link.GetTransform()[0:3, 0:3]
                c_local = link.GetLocalCOM()
                v = link.GetVelocity()
                rd, omega = v[:3], v[3:]
                cd = rd + cross(omega, dot(R, c_local))
                total += m * cd
        return total / self.mass

    def compute_com_jacobian(self, q):
        Jcom = zeros((3, self.nb_dof))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                index = link.GetIndex()
                com = link.GetGlobalCOM()
                m = link.GetMass()
                J = self.rave.ComputeJacobianTranslation(index, com)
                Jcom += m * J
        return Jcom / self.mass

    def compute_com_hessian(self, q):
        Hcom = zeros((self.nb_dof, 3, self.nb_dof))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                index = link.GetIndex()
                com = link.GetGlobalCOM()
                m = link.GetMass()
                H = self.rave.ComputeHessianTranslation(index, com)
                Hcom += m * H
        return Hcom / self.mass

    def compute_com_acceleration(self, q, qd, qdd):
        J = self.compute_com_jacobian(q)
        H = self.compute_com_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qdd))

    def compute_angular_momentum(self, q, qd, p):
        """Compute the angular momentum with respect to point p.

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        momentum = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            for link in self.rave.GetLinks():
                T = link.GetTransform()
                R, r = T[0:3, 0:3], T[0:3, 3]
                c_local = link.GetLocalCOM()  # in local RF
                c = r + dot(R, c_local)

                v = link.GetVelocity()
                rd, omega = v[:3], v[3:]
                cd = rd + cross(omega, dot(R, c_local))

                m = link.GetMass()
                I = link.GetLocalInertia()  # in local RF
                momentum += cross(c - p, m * cd) \
                    + dot(R, dot(I, dot(R.T, omega)))
        return momentum

    def compute_cam(self, q, qd):
        """Compute Centroidal Angular Momentum (CAM), i.e. angular momentum at
        the instantaneous COM."""
        return self.compute_angular_momentum(q, qd, self.compute_com(q))

    def compute_am_pseudo_jacobian(self, q, p):
        """Compute a matrix J(p) such that the angular momentum with respect to
        p is

            L(q, qd) = dot(J(q), qd).

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        J = zeros((3, len(q)))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                i = link.GetIndex()
                c = link.GetGlobalCOM()
                R = link.GetTransform()[0:3, 0:3]
                I = dot(R, dot(link.GetLocalInertia(), R.T))
                J_trans = self.rave.ComputeJacobianTranslation(i, c)
                J_rot = self.rave.ComputeJacobianAxisAngle(i)
                J += dot(crossmat(c - p), m * J_trans) + dot(I, J_rot)
        return J

    def compute_cam_pseudo_jacobian(self, q):
        return self.compute_am_pseudo_jacobian(q, self.compute_com(q))

    def compute_amd_pseudo_hessian(self, q, p):
        """Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to point p is

            Ld(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        where J(q) is the result of self.compute_pseudo_jacobian(q, p).

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        def crosstens(M):
            assert M.shape[0] == 3
            Z = zeros(M.shape[1])
            T = array([[Z, -M[2, :], M[1, :]],
                       [M[2, :], Z, -M[0, :]],
                       [-M[1, :], M[0, :], Z]])
            return T.transpose([2, 0, 1])  # T.shape == (M.shape[1], 3, 3)
        H = zeros((len(q), 3, len(q)))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                i = link.GetIndex()
                c = link.GetGlobalCOM()
                R = link.GetTransform()[0:3, 0:3]
                J_rot = self.rave.ComputeJacobianAxisAngle(i)
                H_trans = self.rave.ComputeHessianTranslation(i, c)
                H_rot = self.rave.ComputeHessianAxisAngle(i)
                I = dot(R, dot(link.GetLocalInertia(), R.T))
                H += middot(crossmat(c - p), m * H_trans) \
                    + middot(I, H_rot) \
                    - dot(crosstens(dot(I, J_rot)), J_rot)
        return H

    def compute_cam_pseudo_hessian(self, q):
        return self.compute_amd_pseudo_hessian(q, self.compute_com(q))

    def compute_cam_rate(self, q, qd, qdd):
        J = self.compute_cam_pseudo_jacobian(q)
        H = self.compute_cam_pseudo_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qd))

    def compute_zmp(self, q, qd, qdd):
        global pb_times, total_times, cum_ratio, avg_ratio
        g = array([0, 0, -9.81])
        f0 = self.mass * g[2]
        tau0 = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            link_velocities = self.rave.GetLinkVelocities()
            link_accelerations = self.rave.GetLinkAccelerations(qdd)
            for link in self.rave.GetLinks():
                mi = link.GetMass()
                ci = link.GetGlobalCOM()
                I_ci = link.GetLocalInertia()
                Ri = link.GetTransform()[0:3, 0:3]
                ri = dot(Ri, link.GetLocalCOM())
                angvel = link_velocities[link.GetIndex()][3:]
                linacc = link_accelerations[link.GetIndex()][:3]
                angacc = link_accelerations[link.GetIndex()][3:]
                ci_ddot = linacc \
                    + cross(angvel, cross(angvel, ri)) \
                    + cross(angacc, ri)
                angmmt = dot(I_ci, angacc) - cross(dot(I_ci, angvel), angvel)
                f0 -= mi * ci_ddot[2]
                tau0 += mi * cross(ci, g - ci_ddot) - dot(Ri, angmmt)
        return cross(array([0, 0, 1]), tau0) * 1. / f0

    def compute_inertia_matrix(self, q, external_torque=None):
        M = zeros((self.nb_dof, self.nb_dof))
        self.rave.SetDOFValues(q)
        for (i, e_i) in enumerate(eye(self.nb_dof)):
            tm, _, _ = self.rave.ComputeInverseDynamics(
                e_i, external_torque, returncomponents=True)
            M[:, i] = tm
        return M

    def display(self):
        self.env.SetViewer('qtcoin')

    def display_com(self, q):
        com = self.compute_com(q)
        display_box(self.env, com, box_id="COM", thickness=0.03)
