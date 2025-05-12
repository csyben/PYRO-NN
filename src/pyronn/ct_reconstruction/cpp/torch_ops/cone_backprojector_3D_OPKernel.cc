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

void Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const float *volume_origin,
                                    const int detector_width, const int detector_height, const float *projection_multiplier);

void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const float *volume_origin,
                                    const int detector_width, const int detector_height, const float *projection_multiplier);



// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ConeBackprojection3D(torch::Tensor sinogram, torch::Tensor volume_shape,
                                torch::Tensor volume_origin, torch::Tensor volume_spacing,
                                torch::Tensor projection_matrices, torch::Tensor projection_multiplier, torch::Tensor hardware_interp ) 
{
  CHECK_INPUT(sinogram);
  CHECK_INPUT(volume_shape);
  CHECK_INPUT(volume_origin);
  CHECK_INPUT(volume_spacing);
  CHECK_INPUT(projection_matrices);
  CHECK_INPUT(projection_multiplier);
  
  auto batch_dim = sinogram.sizes()[0];
  auto out = torch::zeros({batch_dim, volume_shape[0].item<int>(),volume_shape[1].item<int>(), volume_shape[2].item<int>()}, torch::kFloat32).cuda().contiguous();

  for(int index = 0; index < batch_dim; ++index){                                         
    if (!hardware_interp.is_cuda() && hardware_interp[0].item<bool>())
    {
        Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(sinogram[index].data_ptr<float>(), out[index].data_ptr<float>(), projection_matrices.data_ptr<float>(), sinogram.sizes()[1],
                                                        volume_shape[2].item<int>(), volume_shape[1].item<int>(),volume_shape[0].item<int>(), volume_spacing.data_ptr<float>(), volume_origin.data_ptr<float>(),
                                                        sinogram.sizes()[3], sinogram.sizes()[2], projection_multiplier.data_ptr<float>());
    }
    else
    {
        Cone_Backprojection3D_Kernel_Launcher(sinogram[index].data_ptr<float>(), out[index].data_ptr<float>(), projection_matrices.data_ptr<float>(), sinogram.sizes()[1],
                                                        volume_shape[2].item<int>(), volume_shape[1].item<int>(),volume_shape[0].item<int>(), volume_spacing.data_ptr<float>(), volume_origin.data_ptr<float>(),
                                                        sinogram.sizes()[3], sinogram.sizes()[2], projection_multiplier.data_ptr<float>());
    }
  }

 
                                      
  return out;                                    
}