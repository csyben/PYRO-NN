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
 * Links the fan-beam projector layer from python to the actual kernel implementation. Implemented according to Tensorflow API.
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace tensorflow; // NOLINT(build/namespaces)
#include <cuda_runtime_api.h>
#include <cuda.h>

#define CUDA_OPERATOR_KERNEL "FanProjection2D"

REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("volume: float")
    .Input("projection_shape: int32")
    .Input("volume_origin : float")
    .Input("detector_origin : float")
    .Input("volume_spacing : float")
    .Input("detector_spacing : float")
    .Input("source_2_isocenter_distance : float")
    .Input("source_2_detector_distance : float")
    .Input("central_ray_vectors : float")
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
    .Output("output: float")
    .Doc(R"doc(
Computes the 2D fan forward projection of the input based on the given central ray vectors beta

output: A Tensor.
  output = A_fan * x
)doc");

void Fan_Projection_Kernel_Launcher(const float *volume_ptr, float *out, const float *ray_vectors,
                                    const int number_of_projections, const int volume_size_x, const int volume_size_y,
                                    const float *volume_spacing_ptr, const float *volume_origin_ptr,
                                    const int detector_size, const float *detector_spacing, const float *detector_origin, 
                                    const float *sid, const float *sdd);

class FanProjection2DOp : public OpKernel
{
  public:
    explicit FanProjection2DOp(OpKernelConstruction *context) : OpKernel(context){}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat_outer_dims<float>();
        TensorShape input_shape = input_tensor.shape();
        int batch_size = input_tensor.shape().dim_size(0);

         // Grab the projection_shape Tensor.
        const Tensor &input_projection_shape = context->input(1);

        // Grab the volume_origin Tensor. 
        const Tensor &input_volume_origin = context->input(2);
        auto volume_origin = input_volume_origin.flat_outer_dims<float>();
       
        // Grab the detector origin Tensor.  
        const Tensor &input_detector_origin = context->input(3);
        auto detector_origin = input_detector_origin.flat_outer_dims<float>();

        // Grab the volumen_spacing Tensor.  
        const Tensor &input_volume_spacing = context->input(4);
        auto volume_spacing = input_volume_spacing.flat_outer_dims<float>();
        
        // Grab the detector_spacing Tensor.  
        const Tensor &input_detector_spacing = context->input(5);
        auto detector_spacing = input_detector_spacing.flat_outer_dims<float>();
                
        // Grab the detector_spacing Tensor.  
        const Tensor &input_source_isocenter_distance = context->input(6);
        auto sid = input_source_isocenter_distance.flat_outer_dims<float>();
        
        // Grab the detector_spacing Tensor.  
        const Tensor &input_source_detector_distance = context->input(7);
        auto sdd = input_source_detector_distance.flat_outer_dims<float>();
        
        // Grab the ray_vector Tensor.  Assuming input Tensor with [batch, ray_vectors]
        const Tensor &input_ray_vector = context->input(8);
        auto ray_vectors = input_ray_vector.flat_outer_dims<float>();
        TensorShape ray_vector_shape = input_ray_vector.shape();

        // Copy information on output shape to host memory.
        int sino_shape_host_ptr[2];
        auto code = cudaMemcpy(&sino_shape_host_ptr[0], input_projection_shape.flat<int>().data(), 2*sizeof(int), cudaMemcpyDeviceToHost);
        int detector_width =  sino_shape_host_ptr[ 1 ];
        int number_of_projections = sino_shape_host_ptr[ 0 ];

        // Create output shape: [batch_size, projection_shape, detector_width] <--- This is the reason why we need consistent shapes in the batch 
        TensorShape out_shape = TensorShape({batch_size, number_of_projections, detector_width});

        // Create an output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                         &output_tensor));
        auto output = output_tensor->flat_outer_dims<float>();

        // Call the cuda kernel launcher
        for(int index = 0; index < batch_size; ++index){
          Fan_Projection_Kernel_Launcher(&input(index,0), &output(index,0), &ray_vectors(index,0), number_of_projections,
                                        input_shape.dim_size(2), input_shape.dim_size(1), &volume_spacing(index,0), 
                                        &volume_origin(index,0), detector_width,  &detector_spacing(index,0), &detector_origin(index,0),
                                        &sid(index,0), &sdd(index,0));
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), FanProjection2DOp);