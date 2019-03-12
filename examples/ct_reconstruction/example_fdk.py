import numpy as np
import tensorflow as tf
import math

import sys
sys.path.append('/home/markus/FAU/Project/deep_ct_reconstruction_master')

# TODO: better imports
from pyronn.ct_reconstruction.helpers.misc import generate_sinogram
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_3D

import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights

import pyconrad as pyc
pyc.setup_pyconrad()
pyc.start_gui()


def medphys_plot():
    import matplotlib.pyplot as plt
    path = '/home/syben/Documents/medphys/'
    _ = pyc.ClassGetter()
    phantom = pyc.PyGrid.from_tiff(path+'phantom.tif')
    custom_ops = pyc.PyGrid.from_tiff(path + 'short_fdk_custom_ops_reco.tif')
    conrad = pyc.PyGrid.from_tiff(path + 'short_fdk_conrad_reco.tif')

    slice = 128
    row = 128

    phantom_central = np.array(phantom[slice, :, :])
    custom_ops_central = np.array(custom_ops[slice, :, :])
    conrad_central = np.array(conrad[slice, :, :])

    gt = phantom_central[row,:]
    p1 = custom_ops_central[row,:]
    p2 = conrad_central[row,:]

    fig = plt.figure(figsize=(30, 10))
    number_of_collumns = 6
    number_of_rows = 3
    img_one = plt.subplot2grid((number_of_rows, number_of_collumns), (0, 0), colspan=2,rowspan=3)

    plot_one = plt.subplot2grid((number_of_rows, number_of_collumns), (0, 2), colspan=4, rowspan=3)  # , sharex=img_one)


    img_width = np.shape(phantom_central)[0]
    img_one.plot((0, img_width), (row, row), '--', linewidth=5)

    img_one.imshow(phantom_central, cmap=plt.get_cmap('gist_gray'), vmin=0, vmax=0.4)
    img_one.axis('off')
    img_one.set_title('Central Slice NN Reconstruction', fontsize=35, y=1.05)


    plot_one.plot(np.arange(len(gt)), gt, color='#1f77b4')
    plot_one.plot(np.arange(len(p1)), p1, color='lightgreen')
    plot_one.plot(np.arange(len(p2)), p2, linestyle=':', color='red')

    min_plt = 0
    max_plt = 1.1

    plot_one.set_ylim(plt.ylim((min_plt, max_plt)))
    plot_one.legend(['Shepp-Logan Phantom', 'NN Reconstruction', 'CONRAD Reconstruction'], loc='upper center', prop={'size': 35})
    #plt.savefig('evaluation/eval_img/' + experiment_img_name + '.png', dpi=150, transparent=False, bbox_inches='tight')
    plt.gcf().tight_layout(h_pad=0, w_pad=0)

    plt.savefig(path + 'fdk_eval.png', dpi=150, transparent=False, bbox_inches='tight')

    fig.show()

    a = 5

class nn_model:
    def __init__(self, geometry):
        self.geometry = geometry

        self.cosine_weight = tf.get_variable(name='cosine_weight', dtype=tf.float32,
                                             initializer=ct_weights.cosine_weights_3d(self.geometry), trainable=False)

        primary_angles_2 = np.linspace(0, geometry.angular_range, geometry.number_of_projections)

        self.redundancy_weight = tf.get_variable(name='redundancy_weight', dtype=tf.float32,
                                             initializer=ct_weights.parker_weights_3d(self.geometry), trainable=False)

        self.filter = tf.get_variable(name='reco_filter', dtype=tf.float32, initializer=ram_lak_3D(self.geometry), trainable=False)



    def model(self, sinogram):
        self.sinogram_cos = tf.multiply(sinogram, self.cosine_weight)
        self.redundancy_weighted_sino = tf.multiply(self.sinogram_cos,self.redundancy_weight)

        self.weighted_sino_fft = tf.fft(tf.cast(self.redundancy_weighted_sino, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.weighted_sino_fft, tf.cast(self.filter,dtype=tf.complex64))
        self.filtered_sinogram = tf.real(tf.ifft(self.filtered_sinogram_fft))

        self.reconstruction = cone_backprojection3d(self.filtered_sinogram,self.geometry, hardware_interp=True)

        return self.reconstruction, self.redundancy_weighted_sino


def example_cone_3d():
    # ------------------ Declare Parameters ------------------

    # Volume Parameters:
    volume_size = 151
    volume_shape = [volume_size, volume_size, volume_size]
    v_spacing = 0.25
    volume_spacing = [v_spacing,v_spacing,v_spacing]

    # Detector Parameters:
    detector_shape = [251 , 251]
    d_spacing = 0.33
    detector_spacing = [d_spacing,d_spacing]

    # Trajectory Parameters:
    number_of_projections = 248
    angular_range = np.pi+2*np.arctan(detector_shape[0] / 2 / 1200)

    source_detector_distance = 1200
    source_isocenter_distance = 750

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    geometry.angular_range = np.radians(200) #np.pi + 2*geometry.fan_angle
    projection_geometry = circular_trajectory.circular_trajectory_3d(geometry)

    geometry.set_projection_matrices(projection_geometry)
    #geometry.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geometry))
    #print(geometry.projection_matrices[0])
    # Get Phantom 3d
    phantom = shepp_logan.shepp_logan_3d(volume_shape)
    pyc.imshow(phantom, 'phantom')



    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        sinogram = generate_sinogram.generate_sinogram(phantom, cone_projection3d, geometry)
        pyc.imshow(sinogram, 'sinogram')

        model = nn_model(geometry)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        reco_tf, redundancy_weighted_sino_tf = model.model(sinogram)
        reco, redundancy_weighted_sino = sess.run([reco_tf, redundancy_weighted_sino_tf])
    pyc.imshow(reco,'reco')
    #pyc.imshow(redundancy_weighted_sino, 'redundancy_weighted_sino')
    a=5




if __name__ == '__main__':
    #medphys_plot()
    example_cone_3d()
