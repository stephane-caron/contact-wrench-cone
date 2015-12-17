#!/usr/local/bin/sage -python
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

import TOPP
import cvxopt
import cvxopt.solvers
import math
import openravepy
import pickle
import pylab
import pymanoid_sage as pymanoid
import time

from numpy import array, dot, zeros, arange, hstack, cross
from numpy import eye, vstack
from pylab import minimum, maximum, ones
from sage.all import Polyhedron as SagePolyhedron
from sage.all import RDF as SageRDF

from pymanoid_sage import Trajectory
from pymanoid_sage import interpolate
from pymanoid_sage.rotation import quat_from_rpy
from pymanoid_sage.trajectory import FunctionalChunk
from pymanoid_sage.trajectory import PolynomialChunk


env = openravepy.Environment()
env.Load('env.xml')

foot_support = pymanoid.RaveRobotModel(env, 'foot_support')
foot_support.body = foot_support.rave.GetLink("FOOT_SUPPORT_BODY")
foot_support.set_dof_values([0., 0.31, 0.06, math.pi / 8, 0, 0])
foot_support_origin = foot_support.body.GetTransformPose()[4:]

arm_support = pymanoid.RaveRobotModel(env, 'arm_support')
arm_support.body = arm_support.rave.GetLink("ARM_SUPPORT_BODY")
arm_support.set_dof_values([0.3, -0.4, 0.87, 0, 0, 0])
arm_support_origin = arm_support.body.GetTransformPose()[4:]

box = pymanoid.RaveRobotModel(env, 'box')
box.body = box.rave.GetLink("BOX_BODY")
box.set_dof_values([0.45, 0., 0., 0., 0., 0.])
box_origin = box.body.GetTransformPose()[4:]

hrp = pymanoid.HRP4(env)
left_foot = hrp.left_foot
right_foot = hrp.right_foot
left_arm = hrp.left_arm
right_arm = hrp.right_arm

kin_dt = 5e-3
topp_inst = None
DISABLE_RETIMING = False
FOOT_X = 112e-3  # foot half-length
FOOT_Y = 65e-3   # foot half-width
ARM_X = 15e-3    # surface length = 3 cm
ARM_Y = 25e-3    # surface width = 5 cm
mu = 0.7
TOPP_ndiscrsteps = 100

surf_scale = 0.7
FOOT_X *= math.sqrt(surf_scale)
FOOT_Y *= math.sqrt(surf_scale)
ARM_X *= math.sqrt(surf_scale)
ARM_Y *= math.sqrt(surf_scale)

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

DOF_SCALE = 0.95  # additional scaling to avoid joint-limit saturation

RIGHT_FOOT_HALF_SIT = array([
    9.99933956e-01,  -1.05589370e-02,   4.53782388e-03, -1.22760249e-05,
    2.16383058e-02,  -7.71253796e-02, 9.81822260e-02])

RIGHT_ARM_SUPPORT_POSE = hstack([
    quat_from_rpy(1., -math.pi / 2, -math.pi / 2),
    arm_support_origin + array([-0.04, 0.04, 0.08])])

LEFT_FOOT_SUPPORT_POSE = hstack([
    quat_from_rpy(math.pi / 8 + 0.08, 0., 0.),
    foot_support_origin + array([0.0, -0.085,  0.06])])

RIGHT_FOOT_BOX_POSE = hstack([
    quat_from_rpy(0., 0.1, 0.),
    box_origin + array([-0.08 - 0.05, -0.05, 0.19])])

LEFT_FOOT_BOX_POSE = hstack([
    quat_from_rpy(0., 0., 0.),
    box_origin + array([-0.1 - 0.05, +0.13, 0.17])])


def crossmat(x):
    return array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])


def display_green_box(p):
    pymanoid.rave.display_box(hrp.env, p, box_id="Green", thickness=0.03,
                              color='g')


def display_blue_box(p):
    pymanoid.rave.display_box(hrp.env, p, box_id="Blue", thickness=0.03,
                              color='b')


def plot_lists(lists, labels=None):
    from sage.all import list_plot
    c = ['purple', 'green', 'orange']
    if labels:
        p = [list_plot([Ld[i] for Ld in lists], plotjoined=True, color=c[i],
                       legend_label=labels[i]) for i in xrange(3)]
    else:
        p = [list_plot([Ld[i] for Ld in lists], plotjoined=True, color=c[i])
             for i in xrange(3)]
    G = p[0] + p[1] + p[2]
    G.show()


def get_dof_limits(hrp):
    q_min0 = hrp.dof_llim.copy()
    q_max0 = hrp.dof_ulim.copy()
    q_max0[33] = -0.1  # R_SHOULDER_R
    #q_max0[44] = -0.1  # L_ELBOW_P
    q_min0[42] = +0.1  # L_SHOULDER_R
    q_avg = .5 * (q_max0 + q_min0)
    q_dev = .5 * (q_max0 - q_min0)
    q_max = q_avg + DOF_SCALE * q_dev
    q_min = q_avg - DOF_SCALE * q_dev
    return q_max, q_min


class KinTracker(object):

    def __init__(self, vel_fun, jacobian_fun, hessian_fun=None, gain=None):
        self.vel = vel_fun
        self.gain = gain
        self.jacobian = jacobian_fun
        self.hessian_term = hessian_fun


def track_qd(hrp, q0, duration, objectives, constraints, w_reg=1e-3):
    q_max, q_min = get_dof_limits(hrp)
    I = eye(hrp.nb_dof)
    chunks = []
    dt = kin_dt
    q = q0

    qd_max_cst = +1. * ones(len(q))
    qd_min_cst = -1. * ones(len(q))
    #qd_prev = zeros(len(q))

    for x in objectives:
        (w_obj, objective) = x if type(x) is tuple else (1., x)  # krooon
        if w_obj < w_reg:
            print "Warning: w_obj=%f < w_reg=%f" % (w_obj, w_reg)

    for t in arange(0, duration, dt):
        qd_max = minimum(qd_max_cst, 10. * (q_max - q))
        qd_min = maximum(qd_min_cst, 10. * (q_min - q))
        J_list = [c.jacobian(q) for c in constraints]
        v_list = [c.vel(t) for c in constraints]

        P0 = w_reg * I
        q0 = zeros(len(q))

        for x in objectives:
            (w_obj, objective) = x if type(x) is tuple else (1., x)  # krooon
            J = objective.jacobian(q)
            v = objective.vel(t)
            P0 += w_obj * dot(J.T, J)
            q0 += w_obj * dot(-v.T, J)
        qp_P = cvxopt.matrix(P0)
        qp_q = cvxopt.matrix(q0)
        qp_G = cvxopt.matrix(vstack([+I, -I]))
        qp_h = cvxopt.matrix(hstack([qd_max, -qd_min]))
        qp_A = cvxopt.matrix(vstack(J_list))
        qp_b = cvxopt.matrix(hstack(v_list))
        qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
        qd = array(qp_x).reshape((I.shape[0],))

        #qdd = (qd - qd_prev) / dt
        #q = q + qd * dt + .5 * qdd * dt ** 2
        #qd_prev = qd
        #chunk_poly = PolynomialChunk.from_coeffs([.5 * qdd, qd, q], dt)
        #hrp.set_dof_values(q)

        q = q + qd * dt
        chunk_poly = PolynomialChunk.from_coeffs([qd, q], dt)
        chunks.append(chunk_poly)

    return Trajectory(chunks)


def compute_qd0(q, objectives, constraints, w_reg):
    q_max, q_min = get_dof_limits(hrp)
    I = eye(hrp.nb_dof)
    qd_max = 10. * (q_max - q)
    qd_min = 10. * (q_min - q)

    J_list = [c.jacobian(q) for c in constraints]
    v_list = [c.vel(q) for c in constraints]

    qp_P = w_reg * I
    qp_q = zeros(len(q))

    for x in objectives:
        (w_obj, objective) = x if type(x) is tuple else (1., x)  # krooon
        J = objective.jacobian(q)
        v = objective.vel(0)
        #g = objective.gain
        qp_P += w_obj * dot(J.T, J)
        qp_q += w_obj * dot(-v.T, J)
    qp_P = cvxopt.matrix(qp_P)
    qp_q = cvxopt.matrix(qp_q)
    qp_G = cvxopt.matrix(vstack([+I, -I]))
    qp_h = cvxopt.matrix(hstack([qd_max, -qd_min]))
    qp_A = cvxopt.matrix(vstack(J_list))
    qp_b = cvxopt.matrix(hstack(v_list))
    qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
    qd = array(qp_x).reshape((I.shape[0],))
    return qd


def track(hrp, q0, qd0, duration, objectives, constraints, w_reg=1e-3):
    q_max, q_min = get_dof_limits(hrp)
    I = eye(hrp.nb_dof)
    chunks = []
    q = q0.copy()
    qd = qd0.copy()
    dt = kin_dt

    if not DISABLE_RETIMING:  # non-zero initial qd0 for TOPP
        qd = compute_qd0(q0, objectives, constraints, w_reg)
        print "Raising pylab.norm(qd0) to", pylab.norm(qd)

    for x in objectives:
        (w_obj, objective) = x if type(x) is tuple else (1., x)  # krooon
        if w_obj < w_reg:
            print "Warning: w_obj=%f < w_reg=%f" % (w_obj, w_reg)

    for t in arange(0, duration, dt):
        qd_max = 10. * (q_max - q)
        qd_min = 10. * (q_min - q)
        qdd_max = 10. * (qd_max - qd)
        qdd_min = 10. * (qd_min - qd)

        #qd_H_qd_list = [-c.hessian_term(q, qd) for c in constraints]
        #J_list = [c.jacobian(q) for c in constraints]
        A_list = []
        b_list = []

        qp_P = w_reg * I
        qp_q = zeros(len(q))

        for x in objectives:
            (w_obj, obj) = x if type(x) is tuple else (1., x)  # krooon
            J = obj.jacobian(q)
            qd_H_qd = obj.hessian_term(q, qd)
            v = obj.vel(t)
            g = obj.gain
            qp_P += w_obj * dot(J.T, J)
            qp_q += w_obj * dot(-(g * (v - dot(J, qd)) - qd_H_qd).T, J)
        for cons in constraints:
            J = cons.jacobian(q)
            qd_H_qd = cons.hessian_term(q, qd)
            v = cons.vel(t)
            g = cons.gain if cons.gain is not None else 20.
            A_list.append(J)
            b_list.append(g * (v - dot(J, qd)) + qd_H_qd)
        qp_P = cvxopt.matrix(qp_P)
        qp_q = cvxopt.matrix(qp_q)
        qp_G = cvxopt.matrix(vstack([+I, -I]))
        qp_h = cvxopt.matrix(hstack([qdd_max, -qdd_min]))
        qp_A = cvxopt.matrix(vstack(A_list))
        qp_b = cvxopt.matrix(hstack(b_list))
        qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
        qdd = array(qp_x).reshape((I.shape[0],))

        qd = qd + qdd * dt
        q = q + qd * dt + .5 * qdd * dt ** 2
        chunk_poly = PolynomialChunk.from_coeffs([.5 * qdd, qd, q], dt)
        chunks.append(chunk_poly)
        hrp.set_dof_values(q)  # needed for *.vel(t)
        hrp.display_com(q)
        hrp.display_floor_com(q)
        #import time
        #time.sleep(5e-3)

    return Trajectory(chunks)


def span_to_face(M):
    P = SagePolyhedron(rays=[M[:, i] for i in range(M.shape[1])],
                       base_ring=SageRDF)
    C = list(P.Hrepresentation())
    m = len(C)
    n = len(C[0].A())
    Cres = zeros((m, n))
    for i in range(m):
        Cres[i, :] = array(-C[i].A())
    return Cres


def face_to_span(M):
    P = SagePolyhedron(ieqs=[[0] + list(-M[i, :]) for i in range(M.shape[0])],
                       base_ring=SageRDF)
    C = list(P.rays())
    m = len(C)
    n = len(C[0])
    Cres = zeros((n, m))
    for i in range(m):
        Cres[:, i] = array(C[i])
    return Cres


def compute_contact_span(link):
    if link in [left_foot, right_foot]:
        X, Y = FOOT_X, FOOT_Y
    elif link in [left_arm, right_arm]:
        X, Y = ARM_X, ARM_Y
    else:
        assert False, link.GetName()

    # Face representation for individual forces (y)
    M_face_y = zeros((16, 12))
    for i in range(4):
        M_face_y[4 * i,     [3 * i, 3 * i + 1, 3 * i + 2]] = [1, 0, -mu]
        M_face_y[4 * i + 1, [3 * i, 3 * i + 1, 3 * i + 2]] = [-1, 0, -mu]
        M_face_y[4 * i + 2, [3 * i, 3 * i + 1, 3 * i + 2]] = [0, 1, -mu]
        M_face_y[4 * i + 3, [3 * i, 3 * i + 1, 3 * i + 2]] = [0, -1, -mu]

    # Span for individual forces
    S0 = face_to_span(M_face_y)

    # Transform from individual forces to contact wrench (x)
    M_x_to_y = zeros((6, 12))
    M_x_to_y[0, :] = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    M_x_to_y[1, :] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    M_x_to_y[2, :] = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    M_x_to_y[3, :] = Y * array([0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1])
    M_x_to_y[4, :] = -X * array([0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1])
    M_x_to_y[5, :] = X * array([0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0]) \
        - Y * array([1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0])

    # Span for contact wrench
    S = dot(M_x_to_y, S0)
    return S


def compute_GI_face(contacting_links):
    #t0 = time.time()
    ncontacts = len(contacting_links)
    spans = [compute_contact_span(link) for link in contacting_links]
    n = sum([span.shape[1] for span in spans])

    # Span for w_all
    H = zeros((ncontacts * 6, n))
    curcol = 0
    for i in range(ncontacts):
        H[6 * i:6 * (i + 1), curcol:curcol + spans[i].shape[1]] = spans[i]
        curcol += spans[i].shape[1]

    # Transform from w_all to w_GI
    AGI = zeros((6, ncontacts * 6))
    for i in range(ncontacts):
        pi = contacting_links[i].GetTransform()[0:3, 3]
        Ri = contacting_links[i].GetTransform()[0:3, 0:3]
        AGI[:3, 6 * i:6 * i + 3] = -Ri
        AGI[3:, 6 * i:6 * i + 3] = -dot(crossmat(pi), Ri)
        AGI[3:, 6 * i + 3:6 * i + 6] = -Ri

    M = dot(AGI, H)        # span for w_GI
    CGI = span_to_face(M)  # face for w_GI
    #print "Compute CGI (%d contacts): %f s" % ( ncontacts, time.time() - t0)
    print "CGI.shape =", CGI.shape
    return CGI


def ComputeConeConstraints(com_traj, cam_traj, CGI, discrtimestep):
    t0 = time.time()
    ndiscrsteps = int((com_traj.duration + 1e-10) / discrtimestep) + 1
    m, g = hrp.mass, array([0, 0, -9.81])
    a, b, c = [], [], []
    for i in range(ndiscrsteps):
        t = i * discrtimestep
        p = com_traj.q(t)
        pd = com_traj.qd(t)
        pdd = com_traj.qdd(t)
        L = cam_traj.q(t)
        Ld = cam_traj.qd(t)
        a.append(dot(CGI, -hstack([m * pd, m * cross(p, pd) + L])))
        b.append(dot(CGI, -hstack([m * pdd, m * cross(p, pdd) + Ld])))
        c.append(dot(CGI, hstack([m * g, m * cross(p, g)])))
    print "ComputeConeConstraints(%d points): %f s" % (
        ndiscrsteps, time.time() - t0)
    return a, b, c


def retime_centroid_trajectory(com_traj, cam_traj, contacting_links):
    global topp_inst
    assert abs(com_traj.duration - cam_traj.duration) < 1e-10
    duration = com_traj.duration
    if DISABLE_RETIMING:
        return (lambda t: t), duration

    vmax = [0]
    CGI = compute_GI_face(contacting_links)
    discrtimestep = duration / TOPP_ndiscrsteps
    print "TOPP discrtimestep =", discrtimestep
    a, b, c = ComputeConeConstraints(
        com_traj, cam_traj, CGI, discrtimestep)
    topp_traj = TOPP.Chunk(duration, [TOPP.Polynomial([0, 1])])
    topp_inst = TOPP.QuadraticConstraints(
        topp_traj, discrtimestep, vmax, a, b, c)

    try:
        t0 = time.time()
        sd_beg, sd_end = 0, 0
        topp_retimed = topp_inst.Reparameterize(sd_beg, sd_end)
        print "TOPP comp. time = %.2f s" % (time.time() - t0)
    finally:
        for s in arange(0., com_traj.duration, com_traj.duration / 10):
            print "TOPP beta(%.1f, 0) =" % s, topp_inst.solver.GetBeta(s, 0)
        pylab.clf()
        topp_inst.PlotProfiles()
        pylab.ylim(0, 42)
        topp_inst.PlotAlphaBeta()
        pylab.savefig("data/TOPP_profile.pdf")
        pylab.clf()
        topp_inst.PlotProfiles()
        topp_inst.PlotAlphaBeta()
        pylab.savefig("data/TOPP_profile-full.pdf")

    print "retimed duration =", topp_retimed.duration
    s = lambda t: topp_retimed.Eval(t)[0]
    return s, topp_retimed.duration


def retime_whole_body_trajectory(traj, contacting_links):
    com_traj = FunctionalChunk(
        duration=traj.duration,
        q_fun=lambda t: hrp.compute_com(traj.q(t)),
        qd_fun=lambda t: hrp.compute_com_velocity(traj.q(t), traj.qd(t)),
        qdd_fun=lambda t: hrp.compute_com_acceleration(traj.q(t), traj.qd(t),
                                                       traj.qdd(t)))
    cam_traj = FunctionalChunk(
        duration=traj.duration,
        q_fun=lambda t: hrp.compute_cam(traj.q(t), traj.qd(t)),
        qd_fun=lambda t: hrp.compute_cam_rate(traj.q(t), traj.qd(t),
                                              traj.qdd(t)))
    s, new_duration = retime_centroid_trajectory(
        com_traj, cam_traj, contacting_links)
    return traj.retime(s, new_duration)


def play_flying_trajectory(hrp, traj, dt=1e-2):
    for t in arange(0, traj.duration, dt):
        q = traj.q(t)
        q[50:] = zeros(6)
        hrp.set_dof_values(q)
        time.sleep(dt)


def play_com_traj(com_traj):
    dt = com_traj.duration / 100
    for t in arange(0, com_traj.duration, dt):
        com = com_traj.q(t)
        pymanoid.rave.display_box(hrp.env, com, 'COM', thickness=0.05)
        time.sleep(dt)


def fixed_am():
    return KinTracker(
        vel_fun=lambda t: zeros(3),
        jacobian_fun=lambda q: hrp.compute_am_jacobian(q, hrp.compute_com(q)))


def plot_topp_profiles(ylim=None):
    pylab.clf()
    topp_inst.PlotProfiles()
    if ylim is not None:
        pylab.ylim(ylim)
    topp_inst.PlotAlphaBeta()
    pylab.savefig("kron.pdf")


class TrajectorySketch(object):

    def __init__(self, q, qd=None):
        self.chunks = []
        self.cur_q = q
        self.cur_qd = qd if qd is not None else zeros(len(q))
        self.contacting_links = []

    @property
    def cur_com(self):
        return hrp.compute_com(self.cur_q)

    def append_chunk(self, duration, objectives, constraints,
                     **kwargs):
        trackers = [c if type(c) is KinTracker else c(self.cur_q)
                    for c in constraints]
        print "Append chunk duration =", duration
        new_chunk = track(hrp, self.cur_q, self.cur_qd, duration, objectives,
                          constraints=trackers, **kwargs)
        self.chunks.append(new_chunk)
        self.cur_q = new_chunk.last_q
        self.cur_qd = new_chunk.last_qd

    def get_trajectory(self):
        return Trajectory(self.chunks)

    def contact_link(self, link):
        self.contacting_links.append(link)

    def free_link(self, link):
        self.contacting_links.remove(link)

    def contact_constraints(self):
        def fixed_link(link, ref_q):
            return KinTracker(
                vel_fun=lambda t: zeros(6),
                jacobian_fun=lambda q: hrp.compute_link_jacobian(link, q),
                hessian_fun=lambda q, qd: dot(
                    qd, dot(hrp.compute_link_hessian(link, q), qd)))
        return [fixed_link(link, self.cur_q) for link in self.contacting_links]

    def get_com_tracker(self, target_com, duration, gain):
        com_traj = interpolate.linear(
            self.cur_com, target_com, duration=duration)

        def J(q):
            display_green_box(target_com)
            return hrp.compute_com_jacobian(q)
        return KinTracker(
            vel_fun=com_traj.qd,
            gain=gain,
            jacobian_fun=J,
            hessian_fun=lambda q, qd: dot(
                qd, dot(hrp.compute_com_hessian(q), qd)))

    def get_link_tracker(self, link, target_pose, duration, gain):
        assert link not in self.contacting_links

        start_pos = hrp.compute_link_pose(link, self.cur_q)[4:]
        link_pos_traj = interpolate.linear(
            start_pos, target_pose[4:], duration=duration)

        def J(q):
            #J = lambda q: hrp.compute_link_pose_jacobian(link, q)
            display_blue_box(target_pose[4:])
            return hrp.compute_link_pose_jacobian(link, q)

        def v(t):
            steering_pose = target_pose - link.GetTransformPose()
            steering_pose[4:] = link_pos_traj.qd(t)
            return steering_pose

        H_disc = lambda q, qd: (dot(J(q + qd * 1e-5) - J(q), qd)) / 1e-5
        return KinTracker(
            jacobian_fun=J,
            hessian_fun=H_disc,
            vel_fun=v,
            gain=gain)

    def move_dof(self, dof_id, dof_target, duration, gain):
        qd_H_qd = zeros(1)
        J = zeros((1, len(self.cur_q)))
        J[0, dof_id] = 1
        dof_init = self.cur_q[dof_id]
        dof_traj = interpolate.linear(array([dof_init]), array([dof_target]))
        dof_tracker = KinTracker(
            jacobian_fun=lambda q: J,
            hessian_fun=lambda q, qd: qd_H_qd,
            vel_fun=dof_traj.qd,
            gain=gain)
        self.append_chunk(
            duration=duration,
            objectives=[dof_tracker],
            constraints=self.contact_constraints())

    def move_com(self, target_com, duration, gain):
        com_tracker = self.get_com_tracker(target_com, duration, gain)
        self.append_chunk(
            duration=duration,
            objectives=[com_tracker],
            constraints=self.contact_constraints())
        #com_tracker.plot_dev(["COMx", "COMy", "COMz"])

    def move_link(self, link, target_pose, duration, gain):
        link_tracker = self.get_link_tracker(link, target_pose, duration, gain)
        self.append_chunk(
            duration=duration,
            objectives=[link_tracker],
            constraints=self.contact_constraints())

    def move_com_link_hard(self, target_com, link, target_pose, duration=None,
                           gain=None, **kwargs):
        com_tracker = self.get_com_tracker(target_com, duration=duration,
                                           gain=gain)
        link_tracker = self.get_link_tracker(link, target_pose, duration, gain)
        self.append_chunk(
            duration=duration,
            objectives=[link_tracker],
            constraints=self.contact_constraints() + [com_tracker],
            **kwargs)

    def move_link_com(self, link, target_pose, target_com, gain=1.,
                      duration=1., w_link=1e-1, w_com=1.):
        com_tracker = self.get_com_tracker(target_com, duration=duration,
                                           gain=gain)
        link_tracker = self.get_link_tracker(link, target_pose, duration, gain)
        self.append_chunk(
            duration=duration,
            objectives=[(w_link, link_tracker), (w_com, com_tracker)],
            constraints=self.contact_constraints())


def fix_wrists(q):
    q[45:48] = 0
    q[36:39] = 0
    return q


def plot_com(traj, dt=1e-2):
    trange = arange(0, traj.duration, dt)
    com_values = [hrp.compute_com(traj.q(t), traj.qd(t)) for t in trange]
    plot_lists(com_values, ["Lx", "Ly", "Lz"])


def plot_cam(traj, dt=1e-2):
    trange = arange(0, traj.duration, dt)
    cam_values = [hrp.compute_cam(traj.q(t), traj.qd(t)) for t in trange]
    plot_lists(cam_values, ["Lx", "Ly", "Lz"])


def segment_1():
    global DISABLE_RETIMING
    DISABLE_RETIMING = True

    sketch = TrajectorySketch(hrp.q_halfsit)
    sketch.contact_link(left_foot)
    sketch.contact_link(right_foot)

    rarm0 = hrp.compute_link_pose(right_arm, sketch.cur_q)[4:]
    com0 = sketch.cur_com

    via_rarm = [
        hstack([
            array([0.71557187, -0.0973444, -0.66274785,  0.19810669]),
            rarm0 + array([0.2, 0., 0.05])]),
        RIGHT_ARM_SUPPORT_POSE]

    via_com = [
        com0 + array([0.05, 0.02, -0.02]),
        com0 + array([0., -0.08, 0.])]

    for (via_point, via_com) in zip(via_rarm, via_com):
        sketch.move_link_com(
            right_arm, via_point, via_com,
            duration=2., gain=10.)

    DISABLE_RETIMING = False
    return sketch.get_trajectory()


def segment_2(q1):
    global DISABLE_RETIMING
    DISABLE_RETIMING = True

    sketch = TrajectorySketch(q1)
    sketch.contact_link(right_foot)
    sketch.contact_link(right_arm)

    lfoot0 = hrp.compute_link_pose(left_foot, q1)[4:]
    com0 = sketch.cur_com

    via_lfoot = [
        hstack([
            quat_from_rpy(0., 0., 0.),
            lfoot0 + array([0., 0.05, 0.08])]),
        LEFT_FOOT_SUPPORT_POSE]

    via_com = [
        com0 + array([-0.03, 0.0,  -0.04]),
        com0 + array([+0.05, 0.04, 0.01])]

    for (via_point, via_com) in zip(via_lfoot, via_com):
        sketch.move_link_com(
            left_foot, via_point, via_com,
            duration=2., gain=10.)

    DISABLE_RETIMING = False
    traj1 = sketch.get_trajectory()
    traj2 = correct_pre_step_pose(traj1.last_q)
    return Trajectory([traj1, traj2])


def correct_pre_step_pose(q):
    """Compensates for the errors of the acceleration tracker."""

    prep_sketch = TrajectorySketch(q)
    prep_sketch.contact_link(right_foot)
    prep_sketch.contact_link(right_arm)
    prep_sketch.contact_link(left_foot)

    prep_sketch.free_link(right_foot)
    prep_sketch.move_link_com(
        right_foot,
        RIGHT_FOOT_HALF_SIT,
        prep_sketch.cur_com + array([0., 0., 0.]),
        duration=1., gain=20.)
    prep_sketch.contact_link(right_foot)

    prep_sketch.free_link(right_arm)
    prep_sketch.move_link_com(
        right_arm,
        RIGHT_ARM_SUPPORT_POSE,
        prep_sketch.cur_com + array([0., 0., 0.]),
        duration=1., gain=20.)
    prep_sketch.contact_link(right_arm)

    prep_sketch.free_link(left_foot)
    prep_sketch.move_link_com(
        left_foot,
        LEFT_FOOT_SUPPORT_POSE,
        prep_sketch.cur_com + array([0., 0., 0.]),
        duration=1., gain=20.)
    prep_sketch.contact_link(left_foot)

    return prep_sketch.get_trajectory()


def test_cam_matrices(traj, dt=1e-2):
    """Function used to test the CAM pseudo-jacobian and pseudo-hessian
    matrices. Provide with a smooth trajectory."""
    L_dev, Ld_dev, Ld_disc, Ld_jac = [], [], [], []
    for t in arange(0., traj.duration, dt):
        q = traj.q(t)
        qd = traj.qd(t)
        qdd = traj.qdd(t)
        L = hrp.compute_cam(q, qd)
        L2 = hrp.compute_cam(traj.q(t + dt), traj.qd(t + dt))
        Ld_disc.append((L2 - L) / dt)
        J_L = hrp.compute_cam_pseudo_jacobian(q)
        H_L = hrp.compute_cam_pseudo_hessian(q)
        L_dev.append(L - dot(J_L, qd))
        Ld_jac.append(dot(J_L, qdd) + dot(qd, dot(H_L, qd)))
        Ld_dev.append(Ld_disc[-1] - Ld_jac[-1])
    plot_lists(L_dev, ["Lx", "Ly", "Lz"])
    plot_lists(Ld_disc, ["Ld_disc_x", "Ld_disc_y", "Ld_disc_z"])
    plot_lists(Ld_jac, ["Ld_jac_x", "Ld_jac_y", "Ld_jac_z"])
    plot_lists(Ld_dev, ["Ld_dev_x", "Ld_dev_y", "Ld_dev_z"])
    hrp.play_trajectory(traj)


def plot_state_norms(traj):
    l = []
    for t in arange(0, traj.duration, traj.duration / 100):
        l.append([pylab.norm(traj.q(t)),
                  pylab.norm(traj.qd(t)),
                  pylab.norm(traj.qdd(t))])
    plot_lists(l, ["|| q ||", "|| qd ||", "|| qdd ||"])


def segment_3(q2):
    global sketch

    global DISABLE_RETIMING
    #DISABLE_RETIMING = True

    duration = 2.
    gain = 3.

    sketch = TrajectorySketch(q2)
    sketch.contact_link(left_foot)
    sketch.contact_link(right_arm)

    rfoot0 = hrp.compute_link_pose(right_foot, sketch.cur_q)[4:]
    com0 = sketch.cur_com

    via_rfoot = [
        hstack([
            quat_from_rpy(0, -0.05, 0),
            rfoot0 + array([0.05, 0., 0.1])]),
        hstack([
            quat_from_rpy(0, -0.1, 0),
            rfoot0 + array([0.2, 0., 0.15])]),
        RIGHT_FOOT_BOX_POSE]

    via_com = [
        com0 + array([0.05,  0.0, 0.025]),
        com0 + array([0.1,  -0.03, 0.04]),
        com0 + array([0.18, -0.05, 0.])]

    assert len(via_rfoot) == len(via_com)
    for (via_point, via_com) in zip(via_rfoot, via_com):
        sketch.move_link_com(
            right_foot, via_point, via_com,
            w_com=1., w_link=5e-2,
            duration=duration, gain=gain)

    return retime_whole_body_trajectory(
        sketch.get_trajectory(), sketch.contacting_links)


def segment_4(q3):
    global DISABLE_RETIMING
    #DISABLE_RETIMING = True

    duration = 2.
    gain = 10.

    sketch = TrajectorySketch(q3)
    sketch.contact_link(right_foot)
    sketch.contact_link(right_arm)
    #quat0 = hrp.compute_link_pose(left_foot, sketch.cur_q)[:4]
    lfoot0 = hrp.compute_link_pose(left_foot, sketch.cur_q)[4:]
    rfoot0 = hrp.compute_link_pose(right_foot, sketch.cur_q)[4:]
    com0 = sketch.cur_com

    via_lfoot = [
        hstack([
            quat_from_rpy(0., 0.8, 0.),
            #lfoot0 + array([0.1, 0., 0.25])]),
            lfoot0 + array([0.1, 0., 0.1])]),
        #LEFT_FOOT_BOX_POSE]
        hstack([
            quat_from_rpy(0., 1., 0.),
            lfoot0 + array([0.1, 0., 0.3])])]

    via_com = [
        com0 + array([0.15,  -0.05,  0.1]),
        #com0 + array([0.18,  -0.05,   0.1])]
        rfoot0 + array([0.05, -0.05,  -rfoot0[2] + com0[2] + 0.1])]

    assert len(via_lfoot) == len(via_com)
    for (via_point, via_com) in zip(via_lfoot, via_com):
        sketch.move_link_com(
            left_foot, via_point, via_com,
            w_com=1., w_link=5e-2,
            duration=duration, gain=gain)

    traj2 = retime_whole_body_trajectory(
        sketch.get_trajectory(), sketch.contacting_links)

    return traj2


def export_segment(traj, fname):
    if False and raw_input("export %s? (y/[n]) " % fname) != 'y':
        return
    traj1 = Trajectory([FunctionalChunk(
        traj.duration,
        q_fun=lambda t: fix_wrists(traj.q(t)),
        qd_fun=lambda t: fix_wrists(traj.qd(t)),
        qdd_fun=lambda t: fix_wrists(traj.qdd(t)))])
    hrp.write_pos_file(traj1, "data/%s" % fname)


if __name__ == "__main__":
    hrp.display()
    cam_transform = array([
        [-0.39082885,  0.40160035, -0.82823304,  1.98112202],
        [0.92030753,  0.1539337, -0.35963658,  0.79765022],
        [-0.0169372, -0.90278545, -0.42975755,  1.49899077],
        [0.,  0.,  0.,  1.]])
    #hrp.env.GetViewer().SetCamera(cam_transform)
    hrp.set_dof_values(hrp.q_halfsit)
    hrp.collision_handle = None

    segments_to_recompute = [1, 2, 3, 4]

    try:
        q1 = pickle.load(open("data/q1.pkl", "r"))
        q2 = pickle.load(open("data/q2.pkl", "r"))
        q3 = pickle.load(open("data/q3.pkl", "r"))
    except:
        pass

    if 1 in segments_to_recompute:
        traj1 = segment_1()
        q1 = traj1.last_q
        export_segment(traj1, "segment_1")
        print "Writing q1.pkl..."
        pickle.dump(q1, open("data/q1.pkl", "w"))

    if 2 in segments_to_recompute:
        traj2 = segment_2(q1)
        q2 = traj2.last_q
        export_segment(traj2, "segment_2")
        print "Writing q2.pkl..."
        pickle.dump(q2, open("data/q2.pkl", "w"))

    if 3 in segments_to_recompute:
        traj3 = segment_3(q2)
        q3 = traj3.last_q
        print "Writing q3.pkl..."
        pickle.dump(q3, open("data/q3.pkl", "w"))

    if 4 in segments_to_recompute:
        traj4 = segment_4(q3)
        if 3 in segments_to_recompute:
            unretimed_traj = Trajectory([traj3.unretimed, traj4.unretimed])
            retimed_traj = Trajectory([traj3, traj4], unretimed=unretimed_traj)
            export_segment(retimed_traj, "segment_3")

    print "\nAll done.\n"

    import IPython
    IPython.embed()
