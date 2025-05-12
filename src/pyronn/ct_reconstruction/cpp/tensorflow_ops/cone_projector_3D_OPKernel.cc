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
 * Implementation partially adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../helper_headers/helper_geometry_cpu.h"
#include "../helper_headers/helper_eigen.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <Eigen/QR>

#include <typeinfo>
using namespace tensorflow; // NOLINT(build/namespaces)

#define CUDA_OPERATOR_KERNEL "ConeProjection3D"
REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("volume: float")
    .Input("projection_shape: int32")
    .Input("volume_origin : float")
    .Input("volume_spacing : float")
    .Input("projection_matrices : float")
    .Input("step_size: float")
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
Computes the 3D cone forward projection of the input based on the given the trajectory

output: A Tensor.
  output = A_cone * x
)doc");

void Cone_Projection_Kernel_Launcher(const float *volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, 
                                    const int detector_width, const int detector_height, const float *step_size);

void Cone_Projection_Kernel_Tex_Interp_Launcher(const float *volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points,
                                                const int number_of_projections, const int volume_width, const int volume_height,
                                                const int volume_depth, const float *volume_spacing,
                                                const int detector_width, const int detector_height, const float *step_size);

class ConeProjection3DOp : public OpKernel
{
    bool hardware_interp;

    Eigen::Tensor<float, 3, Eigen::RowMajor> inv_AR_matrix;
    Eigen::Tensor<float, 2, Eigen::RowMajor> src_points;

  public:
    
    explicit ConeProjection3DOp(OpKernelConstruction *context) : OpKernel(context)
    { 
        OP_REQUIRES_OK(context, context->GetAttr("hardware_interp", &hardware_interp));
    }

    /*
    https://github.com/tensorflow/tensorflow/issues/5902
    tensorflow::GPUBFCAllocator* allocator = new tensorflow::GPUBFCAllocator(0, sizeof(float) * height * width * 3);
    tensorflow::Tensor input_tensor = tensorflow::Tensor(allocator, tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape( { 1, height, width, 3 }));
    <copy output data from program A into the GPU memory allocated by input_tensor using a GPU->GPU copy>


    https://stackoverflow.com/questions/39797095/tensorflow-custom-allocator-and-accessing-data-from-tensor

    */
   void DecomposeProjectionMatrices( const float *projection_matrices, const float *volume_spacing, const float *volume_origin, const int number_of_projections){
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
         * WARNING: to support dynamic projection matrices the decomposition must be performed before each layer evaluation. Current
         *          implementation is CPU based, which means that the matrices are copied from device memory to host memory each evaluation call.
         *          TODO: Matrix decomposition should be done on GPU and non dynamic matrices would allow to decompose matrices during 
         *                construction time.
         * ********************************************************************************************************************************************************************/
        
        // projection matrices stuff
		src_points = Eigen::Tensor<float, 2, Eigen::RowMajor>(number_of_projections, 3);
        inv_AR_matrix = Eigen::Tensor<float, 3, Eigen::RowMajor>(number_of_projections, 3, 3);

        // Copy spacing from device to host and check if sucessful
        float volume_spacing_host_ptr[3];
        {
            auto code = cudaMemcpy(&volume_spacing_host_ptr[0], volume_spacing, 3*sizeof(float), cudaMemcpyDeviceToHost);
            if (code != cudaSuccess) 
            {
                fprintf(stderr,"GPUassert: %s \n", cudaGetErrorString(code));
                exit(code);
            }     
        }   
        // Copy origin from device to host and check if sucessful
        float volume_origin_host_ptr[3];
        {
            auto code = cudaMemcpy(&volume_origin_host_ptr[0], volume_origin, 3*sizeof(float), cudaMemcpyDeviceToHost);
            if (code != cudaSuccess) 
            {
                fprintf(stderr,"GPUassert: %s \n", cudaGetErrorString(code));
                exit(code);
            }
        }
        // Copy projection matrices to host and check if sucessful
        auto matrices_size = number_of_projections * 12 * sizeof(float);
        float * projection_matrices_host_ptr = new float[matrices_size];
        {            
            auto code = cudaMemcpy(projection_matrices_host_ptr, projection_matrices, matrices_size, cudaMemcpyDeviceToHost);
            if (code != cudaSuccess) 
            {
                fprintf(stderr,"GPUassert: %s \n", cudaGetErrorString(code));
                exit(code);
            }
        }
        Eigen::Matrix3f scaling_matrix(3,3);
        scaling_matrix.setZero();
        scaling_matrix(0,0) = 1.0/volume_spacing_host_ptr[2];
        scaling_matrix(1,1) = 1.0/volume_spacing_host_ptr[1];
        scaling_matrix(2,2) = 1.0/volume_spacing_host_ptr[0];
        src_points.setZero();
        inv_AR_matrix.setZero();

        //for each projection
        for (int n = 0; n < number_of_projections; n++)
        {            
            const float* matrix = &(projection_matrices_host_ptr[n*12]);
            Eigen::Matrix<float,3,4,Eigen::RowMajor> proj_mat(3,4);
            proj_mat << matrix[0], matrix[1], matrix[2 ], matrix[3 ],
                        matrix[4], matrix[5], matrix[6 ], matrix[7 ],
                        matrix[8], matrix[9], matrix[10], matrix[11]; 

            auto c = (Geometry::getCameraCenter(proj_mat) * -1).eval();

            src_points(n,0) = -((volume_origin_host_ptr[ 2 ] * scaling_matrix(0,0)) + c(0) * scaling_matrix(0,0));
            src_points(n,1) = -((volume_origin_host_ptr[ 1 ] * scaling_matrix(1,1)) + c(1) * scaling_matrix(1,1));
            src_points(n,2) = -((volume_origin_host_ptr[ 0 ] * scaling_matrix(2,2)) + c(2) * scaling_matrix(2,2));

            Eigen::Matrix<float,3,3, Eigen::RowMajor> inverted_scaled_result = (scaling_matrix * proj_mat.block<3,3>(0,0).inverse()).eval();

            //TODO: dont copy element-wise use Eigen::Map to map eigen::matrix to eigen::tensor
            for(int j = 0; j < inverted_scaled_result.cols();++j){
                for(int i = 0; i < inverted_scaled_result.rows(); ++i){
                    inv_AR_matrix(n,j,i) = inverted_scaled_result(j,i);
                }
            }           
        }
        // free cpu mem
        delete[] projection_matrices_host_ptr;
   }


    void Compute(OpKernelContext *context) override
    {
        // Grab the input Tensor (Volume).  Assuming input Tensor with [batch, height, width]
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat_outer_dims<float>();
        TensorShape input_shape = input_tensor.shape();
        int batch_size = input_tensor.shape().dim_size(0);

        // Grab the projection_shape Tensor.
        const Tensor &input_projection_shape = context->input(1);

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
        int sino_shape_host_ptr[3];
        auto code = cudaMemcpy(&sino_shape_host_ptr[0], input_projection_shape.flat<int>().data(), 3*sizeof(int), cudaMemcpyDeviceToHost);
        int detector_width =  sino_shape_host_ptr[ 2 ];
        int detector_height =  sino_shape_host_ptr[ 1 ];
        int number_of_projections = sino_shape_host_ptr[ 0 ];
        
        // Create output shape: [batch_size, projection_shape, detector_width] <--- This is the reason why we need consistent shapes in the batch 
        TensorShape out_shape = TensorShape({batch_size, number_of_projections, detector_height, detector_width});
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                         &output_tensor));
        
        auto output = output_tensor->flat_outer_dims<float>();

        // Call the cuda kernel launcher
        for(int index = 0; index < batch_size; ++index){
            DecomposeProjectionMatrices(&proj_matrices(index,0), &volume_spacing(index,0), &volume_origin(index,0), number_of_projections);
            if(hardware_interp){
                Cone_Projection_Kernel_Tex_Interp_Launcher(&input(index,0), &output(index,0), inv_AR_matrix.data(), src_points.data(), number_of_projections,
                                            input_shape.dim_size(3), input_shape.dim_size(2), input_shape.dim_size(1), &volume_spacing(index,0),
                                            detector_width, detector_height, &step_size(index,0));
            }
            else{
                // Call the cuda kernel launcher
                Cone_Projection_Kernel_Launcher(&input(index,0), &output(index,0), inv_AR_matrix.data(), src_points.data(), number_of_projections,
                                            input_shape.dim_size(3), input_shape.dim_size(2), input_shape.dim_size(1),&volume_spacing(index,0), 
                                            detector_width, detector_height, &step_size(index,0));
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), ConeProjection3DOp);