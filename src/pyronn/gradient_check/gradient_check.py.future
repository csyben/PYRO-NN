import numpy as np
import torch

from pyronn.ct_reconstruction.layers.torch.projection_2d import ParallelProjection2D
from pyronn.ct_reconstruction.layers.torch.projection_2d import FanProjection2D
from pyronn.ct_reconstruction.layers.torch.projection_3d import ConeProjection3D
from pyronn.ct_reconstruction.layers.torch.backprojection_2d import ParallelBackProjection2D
from pyronn.ct_reconstruction.layers.torch.backprojection_2d import FanBackProjection2D
# from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
# from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D
# from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory

#torch.set_printoptions(profile='full')
#TODO: TO BE DONE.
def example_parallel_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 32
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [50]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 90
    angular_range = np.pi

    # create Geometry class
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape, volume_spacing, 
                                  detector_shape, detector_spacing, 
                                  number_of_projections, angular_range,
                                  trajectory=circular_trajectory.circular_trajectory_2d)

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(),dtype=torch.float32).cuda()
    phantom.require_grad = False
    # Add required batch dimension
    sino = ParallelProjection2D().forward(phantom, **geometry)
    sino.require_grad = False

    def test_func_proj(x):
        return ParallelProjection2D().forward(x, **geometry)
    def test_func_reco(x):
        return ParallelBackProjection2D().forward(x, **geometry)

    #func = ParallelProjection2D().forward
    #grad_func = ParallelBackProjection2D().forward
    jud = check(test_func_proj, test_func_reco, phantom, [1, volume_size, volume_size])
    #jud = check(test_func_reco, sino, [1,number_of_projections, detector_shape[0]])
    print(np.max(jud))

def check(func, grad_func, input, output_shape, analyse=False):
    eps=1e-4
    #numeric
    grad = torch.zeros(output_shape, dtype=torch.float32).cuda()
    for z in range(output_shape[0]):
        for x in range(output_shape[1]):
            for y in range(output_shape[2]):
                temp_up = input.clone()
                temp_down = input.clone()
                temp_up[z, x, y] += eps
                temp_down[z, x, y] -= eps
                error = grad_func(grad[z, x, y]+1) - (func(temp_up) - func(temp_down)) / (2*eps)
                print(error)
                # jac[:,:,:, z, x, y] /= torch.max(jac[:,:,:,z,x,y])
                # nume[:,:,:, z, x, y] /= torch.max(nume[:,:,:, z, x, y])
    
    # graph_jac = torch.sum(jac, axis=[3,4,5])
    # graph_nume = torch.sum(nume, axis=[3,4,5])
    #
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(graph_jac.cpu()[0])
    # ax[1].imshow(graph_nume.cpu().detach().numpy()[0])
    # fig.show()
    #
    # #compare
    # jac_n = jac.clone()
    # nume_n = nume.clone()
    # jac_n[jac_n>0] = 1
    # nume_n[nume_n>0] = 1
    #
    # union = jac_n + nume_n
    # res = []
    # res_s = []
    # for z in range(shape[0]):
    #     for x in range(shape[1]):
    #         for y in range(shape[2]):
    #             amount = union[:,:,:,z, x, y][union[:,:,:,z, x, y]>0].shape[0]
    #             diff = union[:,:,:,z, x, y][union[:,:,:,z, x, y]==1].shape[0]
    #             same = union[:,:,:,z, x, y][union[:,:,:,z, x, y]==2].shape[0]
    #             res.append(diff / amount)
    #             res_s.append(same / amount)

    # return res

def example_fan_2d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 32
    volume_shape = [volume_size, volume_size]
    volume_spacing = [1, 1]

    # Detector Parameters:
    detector_shape = [100]
    detector_spacing = [1]

    # Trajectory Parameters:
    number_of_projections = 90
    angular_range = np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape, volume_spacing, 
                                  detector_shape, detector_spacing, 
                                  number_of_projections, angular_range,
                                  source_detector_distance = source_detector_distance,
                                  source_isocenter_distance = source_isocenter_distance,
                                  trajectory=circular_trajectory.circular_trajectory_2d)

    # Get Phantom
    phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
    phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(),dtype=torch.float32).cuda()
    phantom.require_grad = False
    sino = FanProjection2D().forward(phantom, **geometry)
    sino.require_grad = False

    def test_func_proj(x):
        return FanProjection2D().forward(x, **geometry)

    def test_func_reco(x):
        return FanBackProjection2D().forward(x,geometry)

    jud = check(test_func_proj, phantom, [1, volume_size, volume_size])
    #jud = check(test_func_reco, sino, [1,number_of_projections, detector_shape[0]])
    print(jud)

def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 8
    volume_shape = [volume_size, volume_size, volume_size]
    volume_spacing = [1, 1, 1]

    # Detector Parameters:
    detector_shape = [12, 12]
    detector_spacing = [1,1]

    # Trajectory Parameters:
    number_of_projections = 12
    angular_range = np.pi

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_3d(geometry))

    # Get Phantom
    phantom = shepp_logan.shepp_logan_3d(volume_shape).astype(np.float32)
    # Add required batch dimension
    phantom = np.expand_dims(phantom, axis=0)
    sino = ConeProjection3D(phantom,geometry)
    @tf.function
    def test_func_proj(x):
        return ConeProjection3D(x,geometry)

    @tf.function
    def test_func_reco(x):
        return ConeBackprojection3D(x,geometry)

    proj_theoretical, proj_numerical = tf.test.compute_gradient(test_func_proj, [sino])
    reco_theoretical, reco_numerical = tf.test.compute_gradient(test_func_reco, [sino])

if __name__ == '__main__':
    example_parallel_2d()
    #example_fan_2d()
    #example_cone_3d()