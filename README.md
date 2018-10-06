# Centroidal wrench cone

Source code for http://www.roboticsproceedings.org/rss11/p28.pdf

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
