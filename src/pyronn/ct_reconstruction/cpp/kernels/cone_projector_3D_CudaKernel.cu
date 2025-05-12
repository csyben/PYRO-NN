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
 * Ray-driven cone-beam projector CUDA kernel using kernel interpolation
 * Implementation adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/

#include <stdio.h>
#include "../helper_headers/helper_grid.h"
#include "../helper_headers/helper_math.h"

#define CUDART_INF_F __int_as_float(0x7f800000)

#define BLOCKSIZE_X           16
#define BLOCKSIZE_Y           16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {        
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

inline __device__ float interp3D(const float* const_volume_ptr, const float3 point, const uint3 pointer_offset, const uint3 volume_size){

    const int x_f = __float2int_rn( floor(point.x) ); 
    const int y_f = __float2int_rn( floor(point.y) );
    const int z_f = __float2int_rn( floor(point.z) );

    const int x_c = x_f+1;
    const int y_c = y_f+1;
    const int z_c = z_f+1;
     
    uint i000 = z_f*pointer_offset.z + y_f * pointer_offset.y + x_f;
    uint i001 = z_f*pointer_offset.z + y_f * pointer_offset.y + x_c;
    uint i010 = z_f*pointer_offset.z + y_c * pointer_offset.y + x_f;
    uint i011 = z_f*pointer_offset.z + y_c * pointer_offset.y + x_c;
    uint i100 = z_c*pointer_offset.z + y_f * pointer_offset.y + x_f;
    uint i101 = z_c*pointer_offset.z + y_f * pointer_offset.y + x_c;
    uint i110 = z_c*pointer_offset.z + y_c * pointer_offset.y + x_f;
    uint i111 = z_c*pointer_offset.z + y_c * pointer_offset.y + x_c;

    float p000 = ( z_f < 0 || z_f >= volume_size.z || y_f < 0 || y_f >= volume_size.y || x_f < 0 || x_f >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i000 ]);
    float p001 = ( z_f < 0 || z_f >= volume_size.z || y_f < 0 || y_f >= volume_size.y || x_c < 0 || x_c >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i001 ]);
    float p010 = ( z_f < 0 || z_f >= volume_size.z || y_c < 0 || y_c >= volume_size.y || x_f < 0 || x_f >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i010 ]); 
    float p011 = ( z_f < 0 || z_f >= volume_size.z || y_c < 0 || y_c >= volume_size.y || x_c < 0 || x_c >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i011 ]);
    float p100 = ( z_c < 0 || z_c >= volume_size.z || y_f < 0 || y_f >= volume_size.y || x_f < 0 || x_f >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i100 ]);
    float p101 = ( z_c < 0 || z_c >= volume_size.z || y_f < 0 || y_f >= volume_size.y || x_c < 0 || x_c >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i101 ]);
    float p110 = ( z_c < 0 || z_c >= volume_size.z || y_c < 0 || y_c >= volume_size.y || x_f < 0 || x_f >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i110 ]);
    float p111 = ( z_c < 0 || z_c >= volume_size.z || y_c < 0 || y_c >= volume_size.y || x_c < 0 || x_c >= volume_size.x ) ? 0.0f : __ldg(&const_volume_ptr[ i111 ]);

    const float x_d = (point.x - x_f);
    const float y_d = (point.y - y_f); 
    const float z_d = (point.z - z_f);

    const float p00 = __fmaf_rn(p100,z_d,__fmaf_rn(p000,-z_d,p000));
    const float p01 = __fmaf_rn(p101,z_d,__fmaf_rn(p001,-z_d,p001));
    const float p10 = __fmaf_rn(p110,z_d,__fmaf_rn(p010,-z_d,p010));
    const float p11 = __fmaf_rn(p111,z_d,__fmaf_rn(p011,-z_d,p011));

    const float p0 = __fmaf_rn(p10,y_d,__fmaf_rn(p00,-y_d,p00));
    const float p1 = __fmaf_rn(p11,y_d,__fmaf_rn(p01,-y_d,p01));

    const float p = __fmaf_rn(p1,x_d,__fmaf_rn(p0,-x_d,p0));

    return p;
}

inline __device__ float kernel_project3D(const float* volume_ptr, const float3 source_point, const float3 ray_vector,
                                         const float step_size, const uint3 volume_size, const uint3 pointer_offsets)
{
    float pixel = 0.0f;
    // Step 1: compute alpha value at entry and exit point of the volume
    float min_alpha, max_alpha;
    min_alpha = 0;
    max_alpha = CUDART_INF_F;
    
    if (0.0f != ray_vector.x)
    {
        float volume_min_edge_point = 0;
        float volume_max_edge_point = volume_size.x;

        float reci = 1.0f / ray_vector.x;
        float alpha0 = (volume_min_edge_point - source_point.x) * reci;
        float alpha1 = (volume_max_edge_point - source_point.x) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ray_vector.y)
    {
        float volume_min_edge_point = 0;
        float volume_max_edge_point = volume_size.y;

        float reci = 1.0f / ray_vector.y;

        float alpha0 = (volume_min_edge_point - source_point.y) * reci;
        float alpha1 = (volume_max_edge_point - source_point.y) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }

    if (0.0f != ray_vector.z)
    {
        float volume_min_edge_point = 0;
        float volume_max_edge_point = volume_size.z;

        float reci = 1.0f / ray_vector.z;

        float alpha0 = (volume_min_edge_point - source_point.z) * reci;
        float alpha1 = (volume_max_edge_point - source_point.z) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }
    // we start not at the exact entry point
    // => we can be sure to be inside the volume
    min_alpha += step_size * 0.5f;
    // Step 2: Cast ray if it intersects the volume
    // Trapezoidal rule (interpolating function = piecewise linear func)

    float3 point = make_float3(0,0,0);
    // Entrance boundary
    // For the initial interpolated value, only a half stepsize is
    //  considered in the computation.
    if (min_alpha < max_alpha)
    {
        point.x = source_point.x + min_alpha * ray_vector.x;
        point.y = source_point.y + min_alpha * ray_vector.y;
        point.z = source_point.z + min_alpha * ray_vector.z;
        pixel += 0.5f * interp3D(volume_ptr, point, pointer_offsets, volume_size);
        min_alpha += step_size;        
    }

    while (min_alpha < max_alpha)
    {
        point.x = source_point.x + min_alpha * ray_vector.x;
        point.y = source_point.y + min_alpha * ray_vector.y;
        point.z = source_point.z + min_alpha * ray_vector.z;
        pixel += interp3D(volume_ptr, point, pointer_offsets, volume_size);;
        min_alpha += step_size;
    }    
    // Scaling by stepsize;
    pixel *= step_size;

    //Last segment of the line
    if (pixel > 0.0f)
    {   
        pixel -= 0.5f * step_size * interp3D(volume_ptr,  point, pointer_offsets, volume_size);
        min_alpha -= step_size;
        float last_step_size = max_alpha - min_alpha;

        pixel += 0.5f * last_step_size* interp3D(volume_ptr, point, pointer_offsets, volume_size);

        point.x = source_point.x + max_alpha * ray_vector.x;
        point.y = source_point.y + max_alpha * ray_vector.y;
        point.z = source_point.z + max_alpha * ray_vector.z;

        // The last segment of the line integral takes care of the
        // varying length.
        pixel += 0.5f * last_step_size * interp3D(volume_ptr, point , pointer_offsets, volume_size);
    }
    return pixel;
}
__global__ void project_3Dcone_beam_kernel( const float* volume_ptr, float *pSinogram, 
                                            const float *d_inv_AR_matrices, const float3 *d_src_points, const float *sampling_step_size,
                                            const uint3 volume_size, const float *volume_spacing_ptr, const uint2 detector_size, const int number_of_projections, 
                                            const uint3 pointer_offsets)
{
    //return;
    uint2 detector_idx = make_uint2( blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y  );
    uint projection_number = blockIdx.z;
    //Prep: Wrap pointer to float2 for better readable code
    float3 volume_spacing = make_float3(*(volume_spacing_ptr+2), *(volume_spacing_ptr+1), *volume_spacing_ptr);
    if (detector_idx.x >= detector_size.x || detector_idx.y >= detector_size.y || blockIdx.z >= number_of_projections)
    {
        return;
    }
    // //Preparations:
	d_inv_AR_matrices += projection_number * 9;
    float3 source_point = d_src_points[projection_number];
    //Compute ray direction
    const float rx = d_inv_AR_matrices[2] + detector_idx.y * d_inv_AR_matrices[1] + detector_idx.x * d_inv_AR_matrices[0];
    const float ry = d_inv_AR_matrices[5] + detector_idx.y * d_inv_AR_matrices[4] + detector_idx.x * d_inv_AR_matrices[3];
    const float rz = d_inv_AR_matrices[8] + detector_idx.y * d_inv_AR_matrices[7] + detector_idx.x * d_inv_AR_matrices[6];

    float3 ray_vector = make_float3(rx,ry,rz);
    ray_vector = normalize(ray_vector);

    float pixel = kernel_project3D(
        volume_ptr,
        source_point,
        ray_vector,
        *sampling_step_size,
        volume_size,
        pointer_offsets);

    unsigned sinogram_idx = projection_number * detector_size.y * detector_size.x +  detector_idx.y * detector_size.x + detector_idx.x;

     pixel *= sqrt(  (ray_vector.x * volume_spacing.x) * (ray_vector.x * volume_spacing.x) +
                     (ray_vector.y * volume_spacing.y) * (ray_vector.y * volume_spacing.y) +
                     (ray_vector.z * volume_spacing.z) * (ray_vector.z * volume_spacing.z)  );

    pSinogram[sinogram_idx] = pixel;
    return;
}

void Cone_Projection_Kernel_Launcher(const float* volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float *volume_spacing, const int detector_width, const int detector_height, const float *step_size)
{
    //COPY inv AR matrix to graphics card as float array
    auto matrices_size_b = number_of_projections * 9 * sizeof(float);
    float *d_inv_AR_matrices;
    gpuErrchk(cudaMalloc(&d_inv_AR_matrices, matrices_size_b));
    gpuErrchk(cudaMemcpy(d_inv_AR_matrices, inv_AR_matrix, matrices_size_b, cudaMemcpyHostToDevice));
    //COPY source points to graphics card as float3
    auto src_points_size_b = number_of_projections * sizeof(float3);

    float3 *d_src_points;
    gpuErrchk(cudaMalloc(&d_src_points, src_points_size_b));
    gpuErrchk(cudaMemcpy(d_src_points, src_points, src_points_size_b, cudaMemcpyHostToDevice));

    uint3 volume_size = make_uint3(volume_width, volume_height, volume_depth);
    uint2 detector_size = make_uint2(detector_width, detector_height);
    uint3 pointer_offsets = make_uint3(1,volume_width,volume_width*volume_height);
    
    const dim3 blocksize = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
    const dim3 gridsize = dim3( detector_size.x / blocksize.x + 1, detector_size.y / blocksize.y + 1 , number_of_projections+1);

    project_3Dcone_beam_kernel<<<gridsize, blocksize>>>(volume_ptr, out, d_inv_AR_matrices, d_src_points, step_size,
                                        volume_size,volume_spacing, detector_size,number_of_projections,pointer_offsets);

    cudaDeviceSynchronize();

    // check for errors
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaFree(d_inv_AR_matrices));
    gpuErrchk(cudaFree(d_src_points));
}

