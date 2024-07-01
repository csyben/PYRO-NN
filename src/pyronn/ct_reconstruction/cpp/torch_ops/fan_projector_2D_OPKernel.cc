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

void Fan_Projection_Kernel_Launcher(const float *volume_ptr, float *out, const float *ray_vectors,
                                    const int number_of_projections, const int volume_size_x, const int volume_size_y,
                                    const float *volume_spacing_ptr, const float *volume_origin_ptr,
                                    const int detector_size, const float *detector_spacing_ptr, const float *detector_origin_ptr,
                                    const float *sid_ptr, const float *sdd_ptr);                                                

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor FanProjection2D(torch::Tensor volume, torch::Tensor projection_shape,
                                torch::Tensor volume_origin, torch::Tensor detector_origin,
                                torch::Tensor volume_spacing, torch::Tensor detector_spacing,
                                torch::Tensor source_isocenter_distance, torch::Tensor source_detector_distance,
                                torch::Tensor ray_vectors) 
{
  CHECK_INPUT(volume);
  CHECK_INPUT(projection_shape);
  CHECK_INPUT(volume_origin);
  CHECK_INPUT(detector_origin);
  CHECK_INPUT(volume_spacing);
  CHECK_INPUT(detector_spacing);
  CHECK_INPUT(source_isocenter_distance);
  CHECK_INPUT(source_detector_distance);
  CHECK_INPUT(ray_vectors);
  
  auto batch_dim = volume.sizes()[0];
  auto out = torch::zeros({batch_dim, projection_shape[0].item<int>(), projection_shape[1].item<int>()}, torch::kFloat32).cuda().contiguous();
  for(int index = 0; index < batch_dim; ++index){
    Fan_Projection_Kernel_Launcher(volume[index].data_ptr<float>(), out[index].data_ptr<float>() , ray_vectors.data_ptr<float>(), projection_shape[0].item<int>(), volume.sizes()[2], volume.sizes()[1],
                                   volume_spacing.data_ptr<float>(), volume_origin.data_ptr<float>(), projection_shape[1].item<int>(),  detector_spacing.data_ptr<float>(), detector_origin.data_ptr<float>(),
                                   source_isocenter_distance.data_ptr<float>(), source_detector_distance.data_ptr<float>() );  }     
  return out;                                    
}