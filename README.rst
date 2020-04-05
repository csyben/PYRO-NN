FRAMEWORK
==========

.. image:: https://badge.fury.io/py/pyronn.svg
   :target: https://badge.fury.io/py/pyronn
   :alt: PyPI version



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
With the new pyronn 0.1.0 Tensorflow 2.x will be supported. The default mode for pyronn is eager execution like Tensorflow itself.
Major features in the update are:

- Tensorflow 2.x support
    - Eager execution for all pyronn_layers
    - Keras support
- Batch size support for all pyronn_layers
- The pyronn_layers wheel is now a dependency and will be installed from pip repositories

Planned
=========


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
======
PYRO-NN comes with all relevant helper classes to easily run the projection and back-projection operators within the Tensorflow context.

To use the Layers a geometry object is needed:

.. code-block:: python

    from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D


    volume_size = 256
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = 512
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_par_projections = 360
    angular_range = 2 * np.pi

    # create Geometry class
    par_geometry = GeometryParallel(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_fan_projections, angular_range)

After defining the basic geometry parameters, a trajectory need to be set. The circular_trajectory class computes an idealiyed
circular trajectory for a given geometry. For 2D parallel- and fan-beam geometry a trajectory is described using the central ray vectors.
For 3D cone-beam geometry the trajectory is described with projection matrices.

The trajectory can be calculated and set as follows:

.. code-block:: python

    from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory

    par_geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(par_geometry))

At this point the geometry is fully setup and can be used to create projections and reconstructions.
The Layers just takes the respective input tensor and the geometry object to conduct the projection, reconstruction respectively.
PYRO-NN also provides convinient general way to create sinograms and reconstructions. The generate methods are generalized
and take the input data, the layer to be used and the geometry. The only restriction is that the generation methods are within
the Tensorflow session scope:

.. code-block:: python

    from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
    from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
    from pyronn.ct_reconstruction.helpers.misc import generate_sinogram as sino_helper
    from pyronn.ct_reconstruction.helpers.misc import generate_reco as reco_helper
    from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan

    phantom = shepp_logan.shepp_logan_enhanced(par_geometry.volume_shape)

    sinogram = sino_helper.generate_sinogram(phantom, parallel_projection2d, par_geometry)
    reconstruction = reco_helper.generate_reco(sinogram, parallel_backprojection2d, par_geometry)

In the following the example using the Layers directly is shown (Note that the Layers are within the Tensorflow graph context
and therefore need to be evaluated before the result can be accessed):

.. code-block:: python

    from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
    from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan

    phantom = shepp_logan.shepp_logan_enhanced(par_geometry.volume_shape)

    sinogram = parallel_projection2d(phantom, par_geometry)

Using the PYRO-NN Layers directly registers the respective gradient, thus they can be used as normal Tensorflow Layers within the graph.
For more details checkout the examples which are covering the different geometry and application cases.

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
