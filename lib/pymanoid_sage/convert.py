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


import TOPP

from trajectory import FunctionalChunk, PolynomialChunk, Trajectory


def from_topp(topp_traj):
    return FunctionalChunk(
        duration=topp_traj.duration,
        q_fun=topp_traj.Eval,
        qd_fun=topp_traj.Evald,
        qdd_fun=topp_traj.Evaldd)


def to_topp(traj):
    def chunk_to_topp(chunk, deg=4):
        topp_polynomials = []
        for q_i in chunk.q_polynoms:
            coeffs = list(q_i.coeffs)
            coeffs.reverse()  # TOPP puts weaker coeffs first
            while len(coeffs) < deg:
                coeffs.append(0.)
            topp_polynomials.append(TOPP.Polynomial(coeffs))
        topp_chunk = TOPP.Chunk(chunk.duration, topp_polynomials)
        return topp_chunk

    chunks = []
    if type(traj) is PolynomialChunk:
        chunks = [chunk_to_topp(traj)]
    elif type(traj) is Trajectory:
        chunks = map(chunk_to_topp, traj.chunks)
    return TOPP.PiecewisePolynomialTrajectory(chunks)
