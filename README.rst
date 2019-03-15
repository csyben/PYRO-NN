FRAMEWORK
==========

.. image:: https://badge.fury.io/py/pyronn.svg
   :target: https://badge.fury.io/py/pyronn
   :alt: PyPI version



The python framework for the PYRO-NN layers implemented in (https://github.com/csyben/PYRO-NN-Layers)

PYRO-NN
=========

PYRO-NN brings state-of-the-art reconstruction algorithm to neural networks integrated into Tensorflow.

To use pyronn you need to build the operators from sources or install the provided binaries from
https://github.com/csyben/PYRO-NN-Layers

The publication can be found under (https://frameworkpaper)

Installation
============

Install via pip :

.. code-block:: bash

   pip install pyronn

or if you downloaded this repository (https://github.com/csyben/PYRO-NN) using:

.. code-block:: bash

   pip install -e .

If you encounter a problem during the installation have a look at our wiki: https://github.com/csyben/PYRO-NN/wiki


Changelog
=========

Can be found `CHANGELOG.md <https://github.com/csyben/PYRO-NN/blob/master/CHANGELOG.md>`_.

Usage
=====
You can start with PYRO-NN


Potential Challenges
====================

Memory consumption on the graphics card can be a problem with CT datasets. For the reconstruction operators the input data is passed via a Tensorflow tensor,
which is already allocated on the graphicscard by Tensorflow itself. In fact without any manual configuration Tensorflow will allocate most of
the graphics card memory and handle the memory management internally. This leads to the problem that CUDA malloc calls in the operators itself will allocate
memory outside of the Tensorflow context, which can easily lead to out of memory errors, although the memory is not full.

There exist two ways of dealing with this problem:

1. A convenient way is to reduce the initially allocated memory by Tensorflow itself and allow a memory growth. We suggest to always use this mechanism
to minimize the occurrence of out of memory errors:

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        ...

2. The memory consuming operators like 3D cone-beam projection and back-projection have a so called hardware_interp flag. This means that the
interpolation for both operators are either done by the CUDA texture or based on software interpolation. To use the CUDA texture,
and thus have a fast hardware_interpolation, the input data need to be copied into a new CUDA array, thus consuming the double amount of memory.
In the case of large data or deeper networks it could be favorable to switch to the software interpolation mode. In this case the actual Tensorflow pointer
can directly be used in the kernel without any duplication of the data. The downside is that the interpolation takes nearly 10 times longer.

Note that the hardware interpolation is the default setup for all operators.