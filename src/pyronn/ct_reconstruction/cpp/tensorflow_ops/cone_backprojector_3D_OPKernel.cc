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
 * Links the cone-beam projector layer from python to the actual kernel implementation. Implemented according to Tensorflow API.
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <typeinfo>
using namespace tensorflow; // NOLINT(build/namespaces)
#include <cuda_runtime_api.h>
#include <cuda.h>
#define CUDA_OPERATOR_KERNEL "ConeBackprojection3D"

REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("sinogram: float")
    .Input("volume_shape: int32")
    .Input("volume_origin : float")
    .Input("volume_spacing : float")
    .Input("projection_matrices : float")
    .Input("step_size : float")
    .Input("projection_multiplier : float")
    .Attr("hardware_interp : bool = false")
    .Output("output: float")
    .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
    {
        ::tensorflow::shape_inference::ShapeHandle batch;
        ::tensorflow::shape_inference::ShapeHandle dim;
        ::tensorflow::shape_inference::ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &dim));  
        TF_RETURN_IF_ERROR(c->Subshape(c->input(0),0,1,&batch));
        TF_RETURN_IF_ERROR(c->Concatenate(batch,dim,&out));
        c->set_output( 0,out);
        return Status::OK();
    } )
    .Doc(R"doc(
Computes the 3D cone backprojection of the input sinogram on the given the trajectory

output: A Tensor.
  output = A_cone^T * p
)doc");

void Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const float *volume_origin,
                                    const int detector_width, const int detector_height, const float *projection_multiplier);

void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const float *volume_origin,
                                    const int detector_width, const int detector_height, const float *projection_multiplier);

class ConeBackprojection3DOp : public OpKernel
{
    bool hardware_interp;

  public:
    explicit ConeBackprojection3DOp(OpKernelConstruction *context) : OpKernel(context)
    {     
        OP_REQUIRES_OK(context, context->GetAttr("hardware_interp", &hardware_interp));   
    }

    void Compute(OpKernelContext *context) override
    {
                // Grab the input Tensor (Volume).  Assuming input Tensor with [batch, height, width]
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat_outer_dims<float>();
        TensorShape input_shape = input_tensor.shape();
        int batch_size = input_tensor.shape().dim_size(0);

        // Grab the projection_shape Tensor.
        const Tensor &input_volume_shape = context->input(1);

        // Grab the volume_origin Tensor. 
        const Tensor &input_volume_origin = context->input(2);
        auto volume_origin = input_volume_origin.flat_outer_dims<float>();

        // Grab the volumen_spacing Tensor.  
        const Tensor &input_volume_spacing = context->input(3);
        auto volume_spacing = input_volume_spacing.flat_outer_dims<float>();
        
        // Grab the ray_vector Tensor.  Assuming input Tensor with [batch, ray_vectors]
        const Tensor &input_proj_matrices = context->input(4);
        auto proj_matrices = input_proj_matrices.flat_outer_dims<float>();
        TensorShape proj_matrices_shape = input_proj_matrices.shape();

        // Grab the detector_spacing Tensor.  
        const Tensor &input_step_size = context->input(5);
        auto step_size = input_step_size.flat_outer_dims<float>();

        // Grab the detector_spacing Tensor.  
        const Tensor &input_projection_multiplier = context->input(6);
        auto proj_multiplier = input_projection_multiplier.flat_outer_dims<float>();
           
        // Copy information on output shape to host memory.
        // Implicit assumption that the input shape is consistent over the batch !
        int volume_shape_host_ptr[3];
        auto code = cudaMemcpy(&volume_shape_host_ptr[0], input_volume_shape.flat<int>().data(), 3*sizeof(int), cudaMemcpyDeviceToHost);
        int volume_x =  volume_shape_host_ptr[ 2 ];
        int volume_y =  volume_shape_host_ptr[ 1 ];
        int volume_z = volume_shape_host_ptr[ 0 ];
        // Create output shape: [batch_size, volume_shape] 
        TensorShape out_shape = TensorShape(
          {batch_size, volume_z,volume_y, volume_x});
        // Create an output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                         &output_tensor));
        auto output = output_tensor->flat_outer_dims<float>();

        // Call the cuda kernel launcher
        for(int index = 0; index < batch_size; ++index){
            if(hardware_interp)
            {
                Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(&input(index,0), &output(index,0),&proj_matrices(index,0),
                                            input_shape.dim_size(1),volume_x, volume_y, volume_z, &volume_spacing(index,0),
                                            &volume_origin(index,0), input_shape.dim_size(3), input_shape.dim_size(2),
                                            &proj_multiplier(index,0));
            }
            else
            {
                Cone_Backprojection3D_Kernel_Launcher(&input(index,0), &output(index,0),&proj_matrices(index,0),
                                            input_shape.dim_size(1),volume_x, volume_y, volume_z, &volume_spacing(index,0),
                                            &volume_origin(index,0), input_shape.dim_size(3), input_shape.dim_size(2),
                                            &proj_multiplier(index,0));
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), ConeBackprojection3DOp);