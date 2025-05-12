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

void Cone_Projection_Kernel_Tex_Interp_Launcher(const float* volume_ptr, float *out, const float *inv_AR_matrix,const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const int detector_width, const int detector_height,const float *step_size);

void Cone_Projection_Kernel_Launcher(const float* volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const int detector_width, const int detector_height, const float *step_size);


void DecomposeProjectionMatrices( torch::Tensor *src_points, torch::Tensor * inv_AR_matrix, torch::Tensor projection_matrices, const float *volume_spacing, const float *volume_origin, const int number_of_projections){
    /*********************************************************************************************************************************************************************
     * 
     *  P = [M | -MC] 
     *  M = KR
     * 1. Extract Source Position (C) from P using SVD to calc right null space
     * 2. Calculate M^-1 and multiply with a 3x3 Matrix containing 1/voxel_spacing on the diagonal matrix
     * 3. Put src_points and inv_ar_matrix into the CUDA Kernel, like Cone-Projector from Conrad
     * 
     * WARNING: The following code is not created under memory and runtime performance point of view.
     *          A better conversion from Tensorflow Tensor to Eigen::Tensor and Eigen::Matrizes are probably neccessary !!!!
     * WARNING: to support dynamic projection matrices the decomposition must be performed before each layer evaluation. 
     *          >>> Current implementation is CPU based <<<, which means that the matrices are copied from device memory to host memory each evaluation call.
     *          TODO: Matrix decomposition should be done on GPU and non dynamic matrices would allow to decompose matrices during 
     *                construction time.
     *          Hint: Torch comes with an svd, but gpu impl. is very slow according to the discussion here: https://discuss.pytorch.org/t/torch-svd-is-slow-in-gpu-compared-to-cpu/10770
     *          Fow now, we use CPU version. Torch gets new linalg module which is also boosted by cuSolvers etc. Probably comes with torch 1.9 . For now use old svd on cpu.
     *          Refactoring neccessary with the new torch version
     * ********************************************************************************************************************************************************************/
    torch::Tensor scaling_matrix = torch::zeros({3, 3}, torch::kFloat32);//.cuda().contiguous();    
    scaling_matrix[0][0] = 1.0/volume_spacing[2];
    scaling_matrix[1][1] = 1.0/volume_spacing[1];
    scaling_matrix[2][2] = 1.0/volume_spacing[0];
    //for each projection
    for (int n = 0; n < number_of_projections; n++)
    {     
        // TODO: adjust start dim for bach support
        auto proj_mat = projection_matrices[n];
        auto proj_mat_block = torch::narrow(proj_mat,1,0,3);
        auto svd_out_tuple = torch::svd(proj_mat,false);
        torch::Tensor v = std::get<2>(svd_out_tuple);
        // We need the last collumn of V
        auto c = torch::narrow(v,1,v.sizes()[0]-1,1);
        
        if (c[3].item<float>()<-1e-12 || c[3].item<float>()>1e-12)
            c=c/c[3].item<float>(); // Def:Camera centers are always positive.
        c = c * -1;

        (*src_points)[n][0] = -((volume_origin[ 2 ] * scaling_matrix[0][0].item<float>()) + c[0].item<float>() * scaling_matrix[0][0].item<float>());
        (*src_points)[n][1] = -((volume_origin[ 1 ] * scaling_matrix[1][1].item<float>()) + c[1].item<float>() * scaling_matrix[1][1].item<float>());
        (*src_points)[n][2] = -((volume_origin[ 0 ] * scaling_matrix[2][2].item<float>()) + c[2].item<float>() * scaling_matrix[2][2].item<float>());

        auto inverted_scaled_result = torch::matmul(scaling_matrix , torch::inverse(proj_mat_block));

        (*inv_AR_matrix)[n] = inverted_scaled_result;       
    }
}

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ConeProjection3D(torch::Tensor volume, torch::Tensor projection_shape,
                                torch::Tensor volume_origin, torch::Tensor volume_spacing,
                                torch::Tensor projection_matrices, torch::Tensor step_size, torch::Tensor hardware_interp ) 
{
    CHECK_INPUT(volume);
    CHECK_INPUT(projection_shape);
    CHECK_INPUT(volume_origin);
    CHECK_INPUT(volume_spacing);
    CHECK_INPUT(projection_matrices);
    CHECK_INPUT(step_size);  

    auto batch_dim = volume.sizes()[0];

    auto out = torch::zeros({batch_dim, projection_shape[0].item<int>(),projection_shape[1].item<int>(), projection_shape[2].item<int>()}, torch::kFloat32).cuda().contiguous();
    auto src_points = torch::zeros({projection_shape.clone().cpu()[0].item<int>(), 3}, torch::kFloat32);
    auto inv_AR_matrix = torch::zeros({projection_shape.clone().cpu()[0].item<int>(), 3, 3}, torch::kFloat32);
    DecomposeProjectionMatrices(&src_points, &inv_AR_matrix, projection_matrices.clone().cpu(), volume_spacing.clone().cpu().data_ptr<float>(), volume_origin.clone().cpu().data_ptr<float>(), projection_shape.clone().cpu()[0].item<int>());
    
    //TODO: Batch support
    for(int index = 0; index < batch_dim; ++index){
        
        
        if (!hardware_interp.is_cuda() && hardware_interp[0].item<bool>())
        {
            Cone_Projection_Kernel_Tex_Interp_Launcher( volume[index].data_ptr<float>(), out[index].data_ptr<float>(), inv_AR_matrix.data_ptr<float>(), src_points.data_ptr<float>() , projection_shape[0].item<int>(), 
                                                        volume.sizes()[3], volume.sizes()[2],volume.sizes()[1], volume_spacing.data_ptr<float>(), 
                                                        projection_shape[2].item<int>(), projection_shape[1].item<int>(), step_size.data_ptr<float>());
        }
        else{
            Cone_Projection_Kernel_Launcher(volume[index].data_ptr<float>(), out[index].data_ptr<float>(), inv_AR_matrix.data_ptr<float>(), src_points.data_ptr<float>() , projection_shape[0].item<int>(),
                                            volume.sizes()[3], volume.sizes()[2],volume.sizes()[1], volume_spacing.data_ptr<float>(),
                                            projection_shape[2].item<int>(), projection_shape[1].item<int>(), step_size.data_ptr<float>());
        }
    }
                                        
    return out;                                    
}


