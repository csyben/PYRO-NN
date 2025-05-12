/*
 * Copyright [2019] [Christopher Syben]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Links the parallel-beam back-projector layer from python to the actual kernel implementation. Implemented according to PyTorch API.
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include <torch/extension.h>
#include <iostream>
#include <vector>

// CUDA forward declarations

void Parallel_Backprojection2D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                               const int volume_size_x, const int volume_size_y,
                                               const float *volume_spacing_ptr, const float *volume_origin_ptr,
                                               const int detector_size, const float *detector_spacing_ptr, const float *detector_origin_ptr);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ParallelBackprojection2D(torch::Tensor sinogram, torch::Tensor volume_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor ray_vectors) 
{
  CHECK_INPUT(sinogram);
  CHECK_INPUT(volume_shape);
  CHECK_INPUT(volume_origin);
  CHECK_INPUT(detector_origin);
  CHECK_INPUT(volume_spacing);
  CHECK_INPUT(detector_spacing);
  CHECK_INPUT(ray_vectors);

  // auto input = sinogram.data_ptr<float>();

  auto batch_dim = sinogram.sizes()[0];
  auto out = torch::zeros({batch_dim, volume_shape[0].item<int>(), volume_shape[1].item<int>()}, torch::kFloat32).cuda().contiguous();
  for(int index = 0; index < batch_dim; ++index){
    Parallel_Backprojection2D_Kernel_Launcher(sinogram[index].data_ptr<float>(), out[index].data_ptr<float>(), ray_vectors.data_ptr<float>(), sinogram.sizes()[1], volume_shape[1].item<int>(),  volume_shape[0].item<int>(),
                                                     volume_spacing.data_ptr<float>(), volume_origin.data_ptr<float>(), sinogram.sizes()[2],  detector_spacing.data_ptr<float>(), detector_origin.data_ptr<float>());
  }
  // auto out = torch::zeros({volume_shape[0].item<int>(), volume_shape[1].item<int>()}, torch::kFloat32).cuda().contiguous();

  // Parallel_Backprojection2D_Kernel_Launcher(input, out.data_ptr<float>(), ray_vectors.data_ptr<float>(), sinogram.sizes()[0], volume_shape[1].item<int>(),  volume_shape[0].item<int>(),
  //                                                   volume_spacing.data_ptr<float>(), volume_origin.data_ptr<float>(), sinogram.sizes()[1],  detector_spacing.data_ptr<float>(), detector_origin.data_ptr<float>());
                  

  return out;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("parallel_backprojection2d", &ParallelBackprojection2D, 
//     R"doc(
//     Computes the 2D parallel backprojection of the input sinogram based on the given ray vectors

//     output: A Tensor.
//       output = A^T * p'
//     )doc");
// }