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


import math
import pylab
import TOPP
import time

from numpy import array, dot, zeros, arange, hstack, cross
from pymanoid_lite.cone_duality import face_of_span, span_of_face
from pymanoid_lite.rotation import crossmat
from pymanoid_lite.trajectory import Chunk
from scipy.linalg import block_diag

FOOT_X = 112e-3  # foot half-length
FOOT_Y = 65e-3   # foot half-width
ARM_X = 15e-3    # surface length = 3 cm
ARM_Y = 25e-3    # surface width = 5 cm
mu = 0.7

surf_scale = 0.7
FOOT_X *= math.sqrt(surf_scale)
FOOT_Y *= math.sqrt(surf_scale)
ARM_X *= math.sqrt(surf_scale)
ARM_Y *= math.sqrt(surf_scale)

ROBOT_MASS = 39.  # [kg]


def report(s):
    print '\033[93m%s\033[0m' % s


def get_link_dimensions(link):
    if 'FOOT' in link.name or 'foot' in link.name:
        return FOOT_X, FOOT_Y
    elif 'ELBOW' in link.name:
        return ARM_X, ARM_Y
    assert False, link.name


def compute_contact_span(link):
    X, Y = get_link_dimensions(link)

    # Face representation for individual forces (y)
    M_face_y = zeros((16, 12))
    for i in range(4):
        M_face_y[4 * i,     [3 * i, 3 * i + 1, 3 * i + 2]] = [1, 0, -mu]
        M_face_y[4 * i + 1, [3 * i, 3 * i + 1, 3 * i + 2]] = [-1, 0, -mu]
        M_face_y[4 * i + 2, [3 * i, 3 * i + 1, 3 * i + 2]] = [0, 1, -mu]
        M_face_y[4 * i + 3, [3 * i, 3 * i + 1, 3 * i + 2]] = [0, -1, -mu]

    # Span for individual forces
    S0 = span_of_face(M_face_y)

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
    print "Span shape:", S.shape
    return S


def compute_GI_face(contacting_links, verbose=True):
    t0 = time.time()
    ncontacts = len(contacting_links)
    spans = [compute_contact_span(link) for link in contacting_links]
    n = sum([span.shape[1] for span in spans])

    # Span for w_all
    H = zeros((ncontacts * 6, n))
    curcol = 0
    for i in range(ncontacts):
        H[6 * i:6 * (i + 1), curcol:curcol + spans[i].shape[1]] = spans[i]
        curcol += spans[i].shape[1]
    print "H.shape:", H.shape

    # Transform from w_all to w_GI
    AGI = zeros((6, ncontacts * 6))
    for i in range(ncontacts):
        pi = contacting_links[i].p
        Ri = contacting_links[i].R
        AGI[:3, 6 * i:6 * i + 3] = -Ri
        AGI[3:, 6 * i:6 * i + 3] = -dot(crossmat(pi), Ri)
        AGI[3:, 6 * i + 3:6 * i + 6] = -Ri
    print "AGI.shape:", AGI.shape

    M = dot(AGI, H)        # span for w_GI
    CGI = face_of_span(M)  # face for w_GI
    if verbose:
        report("Compute CGI (%d contacts): %f ms" % (
            ncontacts, 1000 * (time.time() - t0)))
        report("CGI shape: %s" % str(CGI.shape))
    return CGI


def compute_Coulomb_span():
    M_face_y = zeros((4, 3))
    M_face_y[0, :] = [+1, 0, -mu]
    M_face_y[1, :] = [-1, 0, -mu]
    M_face_y[2, :] = [0, +1, -mu]
    M_face_y[3, :] = [0, -1, -mu]
    return span_of_face(M_face_y)


def compute_GI_face_forces(contacting_links):
    t0 = time.time()

    S0 = compute_Coulomb_span()
    print "Coulomb span shape:", S0.shape

    nb_links = len(contacting_links)
    nb_forces = 4 * nb_links
    H = block_diag(*([S0] * nb_forces))
    print "H.shape:", H.shape

    AGI = zeros((6, 3 * nb_forces))
    for i, link in enumerate(contacting_links):
        X, Y = get_link_dimensions(link)
        p, R = link.p, link.R
        a = [[+X, +Y, 0], [+X, -Y, 0], [-X, -Y, 0], [-X, +Y, 0]]
        for j in xrange(4):
            pi = p + dot(R, a[i])
            AGI[:3, 12 * i + 3 * j:12 * i + 3 * (j + 1)] = -R
            AGI[3:, 12 * i + 3 * j:12 * i + 3 * (j + 1)] = -dot(crossmat(pi), R)
    print "AGI.shape:", AGI.shape

    M = dot(AGI, H)        # span for w_GI
    CGI = face_of_span(M)  # face for w_GI
    report("Compute CGI (%d contacts): %f ms" % (
        nb_forces, 1000 * (time.time() - t0)))
    report("CGI shape: %s" % str(CGI.shape))
    return CGI


def compute_com_cam_traj(com_traj, cam_traj, discrtimestep):
    ndiscrsteps = int((com_traj.T + 1e-10) / discrtimestep) + 1
    p_list = [com_traj.q(i * discrtimestep) for i in xrange(ndiscrsteps)]
    pd_list = [com_traj.qd(i * discrtimestep) for i in xrange(ndiscrsteps)]
    pdd_list = [com_traj.qdd(i * discrtimestep) for i in xrange(ndiscrsteps)]
    L_list = [cam_traj.q(i * discrtimestep) for i in xrange(ndiscrsteps)]
    Ld_list = [cam_traj.qd(i * discrtimestep) for i in xrange(ndiscrsteps)]
    return (p_list, pd_list, pdd_list), (L_list, Ld_list)


def compute_cone_constraints(com_traj, cam_traj, CGI, discrtimestep):
    ndiscrsteps = int((com_traj.T + 1e-10) / discrtimestep) + 1
    com_list, cam_list = compute_com_cam_traj(com_traj, cam_traj, discrtimestep)
    p_list, pd_list, pdd_list = com_list
    L_list, Ld_list = cam_list
    m, g = ROBOT_MASS, array([0, 0, -9.81])
    a, b, c = [], [], []
    t0 = time.time()
    for i in xrange(ndiscrsteps):
        L, Ld = L_list[i], Ld_list[i]
        p, pd, pdd = p_list[i], pd_list[i], pdd_list[i]
        a.append(dot(CGI, -hstack([m * pd, m * cross(p, pd) + L])))
        b.append(dot(CGI, -hstack([m * pdd, m * cross(p, pdd) + Ld])))
        c.append(dot(CGI, hstack([m * g, m * cross(p, g)])))
    report("Compute (a, b, c) vectors (%d points): %f ms (%f ms / point)" % (
        ndiscrsteps - 1, 1000 * (time.time() - t0), (time.time() - t0) * 1000. /
        ndiscrsteps))
    return a, b, c


def retime_centroid_trajectory(com_traj, cam_traj, contacting_links,
                               ndiscrsteps):
    global topp_inst
    assert abs(com_traj.T - cam_traj.T) < 1e-10
    T = com_traj.T

    vmax = [0]
    CGI = compute_GI_face(contacting_links)
    discrtimestep = T / ndiscrsteps
    print "TOPP nb points:", ndiscrsteps
    print "TOPP discrtimestep:", discrtimestep
    a, b, c = compute_cone_constraints(
        com_traj, cam_traj, CGI, discrtimestep)
    topp_traj = TOPP.Chunk(T, [TOPP.Polynomial([0, 1])])
    topp_inst = TOPP.QuadraticConstraints(
        topp_traj, discrtimestep, vmax, a, b, c)

    try:
        t0 = time.time()
        sd_beg, sd_end = 0, 0
        topp_retimed = topp_inst.Reparameterize(sd_beg, sd_end)
        report("TOPP comp. time = %.2f ms" % (1000 * (time.time() - t0)))
    finally:
        if False:
            for s in arange(0., com_traj.T, com_traj.T / 10):
                print "TOPP beta(%.1f, 0) =" % s, topp_inst.solver.GetBeta(s, 0)
        if False:
            pylab.clf()
            topp_inst.PlotProfiles()
            pylab.ylim(0, 42)
            topp_inst.PlotAlphaBeta()
            pylab.savefig("retiming.pdf")
            pylab.clf()
            topp_inst.PlotProfiles()
            topp_inst.PlotAlphaBeta()
            pylab.savefig("retiming-full.pdf")
            print "\nSaving profile plots as PDF..."

    # print "Retimed duration: %.4f s" % topp_retimed.duration

    def s(t):
        return topp_retimed.Eval(t)[0]

    def sd(t):
        return topp_retimed.Evald(t)[0]

    def sdd(t):
        return topp_retimed.Evaldd(t)[0]

    return topp_retimed.duration, s, sd, sdd


def retime_whole_body_trajectory(hrp, traj, contacting_links, ndiscrsteps):
    com_traj = Chunk(
        T=traj.T,
        q=lambda t: hrp.compute_com(traj.q(t)),
        qd=lambda t: hrp.compute_com_velocity(traj.q(t), traj.qd(t)),
        qdd=lambda t: hrp.compute_com_acceleration(traj.q(t), traj.qd(t),
                                                   traj.qdd(t)))
    cam_traj = Chunk(
        T=traj.T,
        q=lambda t: hrp.compute_cam(traj.q(t), traj.qd(t)),
        qd=lambda t: hrp.compute_cam_rate(traj.q(t), traj.qd(t), traj.qdd(t)))
    new_duration = traj.T

    new_duration, s, sd, sdd = retime_centroid_trajectory(
        com_traj, cam_traj, contacting_links, ndiscrsteps)

    return traj.retime(new_duration, s, sd, sdd)
