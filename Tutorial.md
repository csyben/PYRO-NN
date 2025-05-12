## Quickstart

### Installation
#### Default installation
```python
pip install dist\pyronn-1.0.0-cp39-cp39-linux_x86_84.whl
```
requirements:
- pytorch==1.12.1 
- tensorflow==2.11.1 
- cuda==11.3 
- cudnn==8.0 
- linux or windows with wsl
- python==3.9

#### Build from source
##### First of all
If you are using a Linux system, you can proceed to the next part and start building. However, if you are using a Windows system, here are some tips for you.

Currently, Pyronn can only be used with PyTorch directly on the Windows system. If you wish to use Pyronn with TensorFlow on the Windows system, you will need to use the Windows Subsystem for Linux (WSL) to create a Linux environment first.
##### Build
If you wish to use the Pyronn package in a different development environment, we recommend building it from source. To customize the package according to your requirements, please follow these instructions:

Open setup.py. If you want to work with Pyronn with Torch, please comment out "'build': CustomBuild". Otherwise, you can comment out "'build_ext': BuildExtension" to delete the Torch part.
Open pyproject.toml. Here, you can set the build-system. Please ensure that the build environment matches your development environment.
Run the command 'python -m build .'. If you don't have the build package, you can run 'pip install build' first.
If everything works fine, you will get a wheel file under the folder 'dist'. You can use pip to install it.

### Play with pyronn
#### Projection & Reconstruction
The basic function of pyronn is the simulation of projection and reconstruction, here are three examples under folder 'pyronn_examples' named 'example_parallel_2d.py', 'example_fan_2d.py' and 'example_cone_2d.py'.

Tips:
1. To change the backend, you need to import pyronn and use the function pyronn.set_backend(), you only need to do this once, the information about the backend will be saved.
2. Please be careful that the input and the output of projection and backprojection are all numpy array.

[//]: # (This section will show you the basic steps of using pyronn layers for projection based on an easy example. The source code of the example is in "pyronn_examles/example_parallel_2d.py", you can find other examples in folder "pyronn_examples".)

[//]: # ()
[//]: # (First of all, we need a sinogram and the geometry of it.)

[//]: # (```python)

[//]: # (from pyronn.ct_reconstruction.geometry.geometry import Geometry)

[//]: # (from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d)

[//]: # ()
[//]: # (volume_size = 256)

[//]: # (volume_shape = [volume_size, volume_size])

[//]: # (volume_spacing = [1, 1])

[//]: # ()
[//]: # (# Detector Parameters:)

[//]: # (detector_shape = [800])

[//]: # (detector_spacing = [1])

[//]: # ()
[//]: # (# Trajectory Parameters:)

[//]: # (number_of_projections = 360)

[//]: # (angular_range = 2 * np.pi)

[//]: # ()
[//]: # (# create Geometry class)

[//]: # (geometry = Geometry&#40;&#41;)

[//]: # (geometry.init_from_parameters&#40;volume_shape=volume_shape,volume_spacing=volume_spacing,)

[//]: # (                            detector_shape=detector_shape,detector_spacing=detector_spacing,)

[//]: # (                            number_of_projections=number_of_projections,angular_range=angular_range,)

[//]: # (                            trajectory=circular_trajectory_2d&#41;)

[//]: # (```)

[//]: # (You must create a instance of `Geometry` to let the pyronn get this geometry information. [Click]&#40;#Geometry&#41; to know more information about `Geometry`.)

[//]: # ()
[//]: # (If you don't have a scanner, you can also use pyronn to do the simulation.)

[//]: # ()
[//]: # (```python)

[//]: # (from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan)

[//]: # (from pyronn.ct_reconstruction.layers.torch.projection_2d import ParallelProjection2D)

[//]: # ()
[//]: # (phantom = shepp_logan.shepp_logan_enhanced&#40;volume_shape&#41;)

[//]: # (# Add required batch dimension)

[//]: # (phantom = torch.tensor&#40;np.expand_dims&#40;phantom, axis=0&#41;.copy&#40;&#41;, dtype=torch.float32&#41;.cuda&#40;&#41;)

[//]: # ()
[//]: # (# ------------------ Call Layers ------------------)

[//]: # (sinogram = ParallelProjection2D&#40;&#41;.forward&#40;phantom, **geometry&#41;)

[//]: # (```)

[//]: # (You must make sure that the geometry for projection and reconstruction is absolutely the same.)

[//]: # (`ParalleProjection2d&#40;&#41;.forward&#40;phantom, **geometry&#41;` will perform as a scanner.)

[//]: # ()
[//]: # (To implement a FBP algorithm, we need to filter the sinogram first.)

[//]: # (```python)

[//]: # (import torch)

[//]: # (from pyronn.ct_reconstruction.helpers.filters import filters)

[//]: # ()
[//]: # (reco_filter = torch.tensor&#40;filters.shepp_logan_2D&#40;geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections&#41;&#41;.cuda&#40;&#41;)

[//]: # (x = torch.fft.fft&#40;sinogram,dim=-1,norm="ortho"&#41;)

[//]: # ()
[//]: # (x = torch.multiply&#40;x,reco_filter&#41;)

[//]: # (x = torch.fft.ifft&#40;x,dim=-1,norm="ortho"&#41;.real)

[//]: # (```)

[//]: # (Please be aware of the shape of your filter. In module `filters` you can find more filters we provide)

[//]: # ()
[//]: # (After this, with only one line, you can get the reconstruction result.)

[//]: # ()
[//]: # (```python)

[//]: # (from pyronn.ct_reconstruction.layers.torch.backprojection_2d import ParallelBackProjection2D)

[//]: # ()
[//]: # (reco = ParallelBackProjection2D&#40;&#41;.forward&#40;x.contiguous&#40;&#41;, **geometry&#41;)

[//]: # ()
[//]: # (reco = reco.cpu&#40;&#41;.numpy&#40;&#41;)

[//]: # (```)

[//]: # (Please be aware that in pyronn, projection and reconstruction are separated into two different class.)

[//]: # (And don't forget to detach it from gpu.)

[//]: # ()
[//]: # (## Geometry)

[//]: # (The `Geometry` class can be initial from parameters, json files or EZRT files. The type of geometry&#40;parallel beam, fan beam and cone beam&#41; )

[//]: # (is depended on the trajectory you provide. )

[//]: # (The properties are saved in a dictionary named parameter_dict. Here's the properties of `Geometry`.)

[//]: # ()
[//]: # (| Keys                      | Description                                                                                           |)

[//]: # (|---------------------------|-------------------------------------------------------------------------------------------------------|)

[//]: # (| volume_shape              | [volume_Z, volume_X, Volume_Y]                                                                        |)

[//]: # (| volume_spacing            | spacing for each axis                                                                                 |)

[//]: # (| volume_origin             | coordinate of the volume origin                                                                       |)

[//]: # (| detector_shape            | [detector_height, detector_width                                                                      |)

[//]: # (| detector_spacing          | spacing for detector                                                                                  |)

[//]: # (| detector_origin           | coordinate of the detector center                                                                     |)

[//]: # (| number_of_projections     | projection amount                                                                                     |)

[//]: # (| angular_range             | can be a 2-elements list or a float value. If only one value is provide, the range will be [0, value] |)

[//]: # (| sinogram_shape            | shape of the sinogram, will be calculate automatically                                                |)

[//]: # (| source_detector_distance  | distance between source and detector, not pixel distance                                              |)

[//]: # (| source_isocenter_distance | distance between source and iso-center, not pixel distance                                            |)

[//]: # (| trajectory                | the result of the trajectory  calculation function                                                    |)

[//]: # (| projection_multiplier     | will be calculated automatically                                                                      |)

[//]: # (| step_size                 | sample step size, default value is 0.2                                                                |)

[//]: # ()
[//]: # (The functions provide by  `Geometry`:)

[//]: # ()
[//]: # (| functions           | Description                                        |)

[//]: # (|---------------------|----------------------------------------------------|)

[//]: # (| fan_angle           | get the trajectory angle values                    |)

[//]: # (| cone_angle          | get the trajectory angle values                    |)

[//]: # (| set_detector_shift  | shift the origin if necessary                      |)

[//]: # (| set_volume_slice    | &#40;this one is not working right now&#41;                |)

[//]: # (| set_angle_range     | modify the project angle                           |)

[//]: # (| swap_axis           | set the direction of the rotation of the system    |)

[//]: # (| slice_the_geometry  | divide the geometry into several smaller geometry  |)



