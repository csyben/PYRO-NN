FRAMEWORK
==========

.. image:: https://badge.fury.io/py/pyronn.svg
   :target: https://badge.fury.io/py/pyronn
   :alt: PyPI version

.. image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: code_of_conduct.md

The Python framework for the PYRO-NN layers implemented in (https://github.com/csyben/PYRO-NN-Layers)

PYRO-NN
=========

PYRO-NN brings state-of-the-art reconstruction algorithm to neural networks integrated into TensorFlow and PyTorch.  
Open access paper available under:  
[https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13753](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13753)

pyronn depends on the pyronn_layers, which are now installed via pip. The source code of the pyronn_layers can be found under:  
[https://github.com/csyben/PYRO-NN-Layers](https://github.com/csyben/PYRO-NN-Layers)

If you find this helpful, we kindly ask you to reference our article published in Medical Physics:

.. code-block:: 

   @article{PYRONN2019,
   author = {Syben, Christopher and Michen, Markus and Stimpel, Bernhard and Seitz, Stephan and Ploner, Stefan and Maier, Andreas K.},
   title = {Technical Note: PYRO-NN: Python reconstruction operators in neural networks},
   year = {2019},
   journal = {Medical Physics},
   }

Documentation
===============
For comprehensive documentation, please visit our website:  
[https://pyronn-doc.github.io/](https://pyronn-doc.github.io/)

Update
=========

Q&A
=========
    - DLL load failed while importing pyronn_layers: Das angegebene Modul wurde nicht gefunden. (import pyronn first)
    - DLL load failed while importing pyronn_layers: Die angegebene Prozedur wurde nicht gefunden. (torch version problem)

Installation
============

Install via pip :

NOT supported remotely yet. Contact maintainers to get the wheel file.  
.. code-block:: bash

   pip install pyronn-0.1.2-*.whl (Not Supported yet)

or you can download this repository (https://github.com/csyben/PYRO-NN) and build a wheel by yourself:

    - Microsoft Visual C++ 14.0 or greater is required
    - For Windows system, WSL2 is required
    - Build package is required
    - CUDA is required (more than 10.2)
    
    1. Clone the repository
    2. Change into torch+tf branch
    3. Modify TOML file if needed; please make sure the torch version in TOML file is the same as your environment, or DLL errors will occur.
    4. Run "python -m build ."
    5. The wheel file will be in the dist directory.

If necessary, you can change `pyproject.toml` for specific torch versions here:

   requires = ["setuptools==69.5.1",
           "ninja",
           "tensorflow[and-cuda]==2.11.1",
           "--extra-index-url https://download.pytorch.org/whl/cu118",
           "torch==2.3.0+cu118",
   ]


If you encounter a problem during installation, have a look at our wiki: [https://github.com/csyben/PYRO-NN/wiki](https://github.com/csyben/PYRO-NN/wiki)

Related Repositories
====================

- [Trainable Bilateral Filter](https://github.com/sypsyp97/trainable_bilateral_filter_torch): A pure PyTorch implementation of trainable bilateral filter for image denoising.
- [Geometry Gradients CT](https://github.com/mareikethies/geometry_gradients_CT): This repository contains the code for computed tomography (CT) reconstruction in fan-beam and cone-beam geometry, which is differentiable with respect to its acquisition geometry.

Potential Challenges
====================

Memory consumption on the graphics card can be a problem with CT datasets. For the reconstruction operators, the input data is passed via a TensorFlow tensor,
which is already allocated on the graphics card by TensorFlow itself. In fact, without any manual configuration, TensorFlow will allocate most of
the graphics card memory and handle the memory management internally. This leads to the problem that CUDA malloc calls in the operators will allocate
memory outside of the TensorFlow context, which can easily lead to out-of-memory errors, although the memory is not full.

There exist two ways of dealing with this problem:

1. With the new pyronn version of 0.1.0, pyronn will automatically set memory growth for TensorFlow to true. The following code allows the memory growth:

.. code-block:: python

    gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

2. The memory-consuming operators like 3D cone-beam projection and back-projection have a so-called `hardware_interp` flag. This means that the
interpolation for both operators is either done by the CUDA texture or based on software interpolation. To use the CUDA texture,
and thus have a fast hardware interpolation, the input data needs to be copied into a new CUDA array, thus consuming double the amount of memory.
In the case of large data or deeper networks, it could be favorable to switch to the software interpolation mode. In this case, the actual TensorFlow pointer
can directly be used in the kernel without any duplication of the data. The downside is that the interpolation takes nearly 10 times longer.

Note that the hardware interpolation is the default setup for all operators.
