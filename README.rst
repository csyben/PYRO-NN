FRAMEWORK
==========

.. image:: https://badge.fury.io/py/pyronn.svg
   :target: https://badge.fury.io/py/pyronn
   :alt: PyPI version

.. image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: code_of_conduct.md

The python framework for the PYRO-NN layers implemented in (https://github.com/csyben/PYRO-NN-Layers)

PYRO-NN
=========

PYRO-NN brings state-of-the-art reconstruction algorithm to neural networks integrated into Tensorflow.
Open access paper available under:
https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13753

pyronn depends on the pyronn_layers. They are now installed via pip. The source code of the pyronn_layers can be found under:
https://github.com/csyben/PYRO-NN-Layers

If you find this helpful, we would kindly ask you to reference our article published in medical physics:

.. code-block:: 

   @article{PYRONN2019,
   author = {Syben, Christopher and Michen, Markus and Stimpel, Bernhard and Seitz, Stephan and Ploner, Stefan and Maier, Andreas K.},
   title = {Technical Note: PYRO-NN: Python reconstruction operators in neural networks},
   year = {2019},
   journal = {Medical Physics},
   }

Update
=========

Q&A
=========
    - DLL load failed while importing pyronn_layers: Das angegebene Modul wurde nicht gefunden. (import pyronn first)
    - DLL load failed while importing pyronn_layers: Die angegebene Prozedur wurde nicht gefunden. (torch version problem)

Installation
============

Install via pip :

NOT support remotely yet. Contact maintainers to get the wheel file.
.. code-block:: bash

   pip install pyronn-0.1.2-*.whl

or you can downloaded this repository (https://github.com/csyben/PYRO-NN) and build a wheel by yourself:

    - Microsoft Visual C++ 14.0 or greater is required
    - for windows system, wsl2 is required
    - build package is required
    - cuda is required(more than 10.2)
    
    1. clone the repository
    2. change into combination branch
    3. modify toml file if needed, please make sure the torch version in toml file is the same as your enviornment, or DLL error will happens.
    4. run "python -m build ."
    5. the wheel file will be at the dist directory.

If necessary, you can change pyproject.toml for specific torch version.

If you encounter a problem during the installation have a look at our wiki: https://github.com/csyben/PYRO-NN/wiki

Related Repositories
====================

- [Trainable Bilateral Filter](https://github.com/sypsyp97/trainable_bilateral_filter_torch): A pure PyTorch implementation of trainable bilateral filter for image denoising.
- [Geometry Gradients CT](https://github.com/mareikethies/geometry_gradients_CT): This repository contains the code for computed tomography (CT) reconstruction in fan-beam and cone-beam geometry which is differentiable with respect to its acquisition geometry.

Potential Challenges
====================

Memory consumption on the graphics card can be a problem with CT datasets. For the reconstruction operators the input data is passed via a Tensorflow tensor,
which is already allocated on the graphicscard by Tensorflow itself. In fact without any manual configuration Tensorflow will allocate most of
the graphics card memory and handle the memory management internally. This leads to the problem that CUDA malloc calls in the operators itself will allocate
memory outside of the Tensorflow context, which can easily lead to out of memory errors, although the memory is not full.

There exist two ways of dealing with this problem:

1. With the new pyronn version of 0.1.0 pyronn will automatically set memory growth for Tensorflow to true. The following code allows the memory growth:

.. code-block:: python

    gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RunetimeError as e:
                print(e)

2. The memory consuming operators like 3D cone-beam projection and back-projection have a so called hardware_interp flag. This means that the
interpolation for both operators are either done by the CUDA texture or based on software interpolation. To use the CUDA texture,
and thus have a fast hardware_interpolation, the input data need to be copied into a new CUDA array, thus consuming the double amount of memory.
In the case of large data or deeper networks it could be favorable to switch to the software interpolation mode. In this case the actual Tensorflow pointer
can directly be used in the kernel without any duplication of the data. The downside is that the interpolation takes nearly 10 times longer.

Note that the hardware interpolation is the default setup for all operators.
