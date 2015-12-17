# Leveraging Cone Double Description for Multi-contact Stability of Humanoids with Applications to Statics and Dynamics

Source code for http://www.roboticsproceedings.org/rss11/p28.pdf

## Abstract

We build on previous works advocating the use of the Gravito-Inertial Wrench
Cone (GIWC) as a general contact stability criterion (a "ZMP for non-coplanar
contacts"). We show how to compute this wrench cone from the friction cones of
contact forces by using an intermediate representation, the surface contact
wrench cone, which is the minimal representation of contact stability for each
surface contact. The observation that the GIWC needs to be computed only once
per stance leads to particularly efficient algorithms, as we illustrate in two
important problems for humanoids : "testing robust static equilibrium" and
"time-optimal path parameterization". We show, through theoretical analysis and
in physical simulations, that our method is more general and/or outperforms
existing ones.

Authors:
[Stéphane Caron](https://scaron.info),
[Quang-Cuong Pham](https://www.normalesup.org/~pham/) and
[Yoshihiko Nakamura](http://www.ynl.t.u-tokyo.ac.jp/)

## Content

- ``box/``: generate the box climbing motion (Section V)
- ``lib/``: various (old) versions of the [pymanoid](https://github.com/stephane-caron/pymanoid) library
- ``perf/``: sample log files and small script to analyse computation times
- ``robust/``: robust static equilibrium criterion (Section IV)
- ``stair/``: retime a stair climbing motion (Section V)

## Requirements

- [CVXOPT](http://cvxopt.org/) (2.7.6)
- [NumPy](http://www.numpy.org/) (1.8.2)
- [OpenRAVE](https://github.com/rdiankov/openrave) (0.9.0)
- [pycddlib](https://github.com/mcmtroffaes/pycddlib) (1.0.5a1)
- [SageMath](http://www.sagemath.org/) (6.9) for the box climbing motion
- [TOPP](https://github.com/quangounet/TOPP/commit/ef1688db4fc49b4dcba98e361696c9caadbe5631)

You will also need the ``HRP4R.dae`` Collada model for HRP4 (md5sum
``dcea527e4fb2e7abae64a27a017102e4`` for our version), as well as the
``hrp4.py`` helper scripts in the library folders. Unfortunately it is unclear
whether we can release these files here due to copyright problems.

## SageMath installation

Link the ``openravepy`` and ``TOPP`` python modules into your SageMath
installation. For example, if your sage directory is in ``~/Software/sage``:

```
cd ~/Software/sage/local/lib/python2.7/site-packages
ln -sf /usr/local/lib/python2.7/dist-packages/openravepy
ln -sf /usr/local/lib/python2.7/dist-packages/TOPP 
```

## TOPP tunings

You may need to set the ``integrationtimestep`` to a smaller value than the one
computed by TOPP. In these experiments, we used ``integrationtimestep=1e-4``.
