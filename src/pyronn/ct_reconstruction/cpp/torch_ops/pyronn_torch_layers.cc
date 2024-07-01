#include <torch/extension.h>

//Forward declaration of the layers
// Parallel operators
torch::Tensor ParallelProjection2D(torch::Tensor volume, torch::Tensor projection_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor ray_vectors);

torch::Tensor ParallelBackprojection2D(torch::Tensor sinogram, torch::Tensor volume_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor ray_vectors) ;

// Fan Operators

torch::Tensor FanProjection2D(torch::Tensor volume, torch::Tensor projection_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor source_isocenter_distance, torch::Tensor source_detector_distance,
                                torch::Tensor ray_vectors) ;

torch::Tensor FanBackprojection2D(torch::Tensor sinogram, torch::Tensor volume_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor source_isocenter_distance, torch::Tensor source_detector_distance,
                                torch::Tensor ray_vectors);

//Cone operators

torch::Tensor ConeProjection3D(torch::Tensor volume, torch::Tensor projection_shape,
                                torch::Tensor volume_origin, torch::Tensor volume_spacing,
                                torch::Tensor projection_matrices, torch::Tensor step_size, torch::Tensor hardware_interp );

torch::Tensor ConeBackprojection3D(torch::Tensor sinogram, torch::Tensor volume_shape,
                                torch::Tensor volume_origin, torch::Tensor volume_spacing,
                                torch::Tensor projection_matrices, torch::Tensor projection_multiplier, torch::Tensor hardware_interp );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //Parallel operators
    m.def("parallel_projection2d", &ParallelProjection2D, 
    R"doc(
    Computes the 2D parallel projection of the input sinogram based on the given ray vectors

    output: A Tensor.
      output = A * p
    )doc");    

    m.def("parallel_backprojection2d", &ParallelBackprojection2D, 
    R"doc(
    Computes the 2D parallel backprojection of the input sinogram based on the given ray vectors

    output: A Tensor.
      output = A^T * p'
    )doc");

    // Fan operators

    m.def("fan_projection2d", &FanProjection2D, 
    R"doc(
    Computes the 2D fan projection of the input sinogram based on the given central ray vectors

    output: A Tensor.
      output = A * p
    )doc");

    m.def("fan_backprojection2d", &FanBackprojection2D, 
    R"doc(
    Computes the 2D fan backprojection of the input sinogram based on the given central ray vectors

    output: A Tensor.
      output = A^T * p'
    )doc");

    m.def("cone_projection3d", &ConeProjection3D, 
    R"doc(
    Computes the 3D cone projection of the input sinogram based on the given trajectory

    output: A Tensor.
      output = A * p
    )doc");


    m.def("cone_backprojection3d", &ConeBackprojection3D, 
    R"doc(
    Computes the 3D cone backprojection of the input sinogram based on the given trajectory

    output: A Tensor.
      output = A^T * p'
    )doc");



}
