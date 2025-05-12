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
 * Voxel-driven cone-beam back-projector CUDA kernel using kernel interpolation
 * Implementation adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include <stdio.h>
#include "../helper_headers/helper_grid.h"
#include "../helper_headers/helper_math.h"

#define BLOCKSIZE_X           16
#define BLOCKSIZE_Y           4
#define BLOCKSIZE_Z           4

texture<float, cudaTextureType2DLayered> sinogram_as_texture;

#define CUDART_INF_F __int_as_float(0x7f800000)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline __device__ float3 map( float3 coordinates, const float* d_projection_matrices, int n )
{
   const float* matrix = &(d_projection_matrices[n*12]);

   return make_float3(
         matrix[0] * coordinates.x + matrix[1] * coordinates.y + matrix[2] * coordinates.z + matrix[3],
         matrix[4] * coordinates.x + matrix[5] * coordinates.y + matrix[6] * coordinates.z + matrix[7],
         matrix[8] * coordinates.x + matrix[9] * coordinates.y + matrix[10] * coordinates.z + matrix[11]
   );
}

inline __device__ float interp2D(const float* const_volume_ptr, const float3 point, const uint3 pointer_offset, const uint2 detector_size){

   const int x_f = __float2int_rn( floor(point.x) ); 
   const int y_f = __float2int_rn( floor(point.y) );

   const int x_c = x_f+1;
   const int y_c = y_f+1;

   int i00 =(uint)point.z*pointer_offset.z + y_f * pointer_offset.y + x_f;
   int i01 =(uint)point.z*pointer_offset.z + y_f * pointer_offset.y + x_c;
   int i10 =(uint)point.z*pointer_offset.z + y_c * pointer_offset.y + x_f;
   int i11 =(uint)point.z*pointer_offset.z + y_c * pointer_offset.y + x_c;

   float p00 = ( x_f < 0 || x_f > detector_size.x || y_f < 0 || y_f > detector_size.y ) ? 0.0f : __ldg(&const_volume_ptr[ i00 ]);
   float p01 = ( x_c < 0 || x_c > detector_size.x || y_f < 0 || y_f > detector_size.y ) ? 0.0f : __ldg(&const_volume_ptr[ i01 ]);
   float p10 = ( x_f < 0 || x_f > detector_size.x || y_c < 0 || y_c > detector_size.y ) ? 0.0f : __ldg(&const_volume_ptr[ i10 ]); 
   float p11 = ( x_c < 0 || x_c > detector_size.x || y_c < 0 || y_c > detector_size.y ) ? 0.0f : __ldg(&const_volume_ptr[ i11 ]);

   const float x_d = (point.x - x_f);
   const float y_d = (point.y - y_f); 

   const float p0 = __fmaf_rn(p01,x_d,__fmaf_rn(p00,-x_d,p00));
   const float p1 = __fmaf_rn(p11,x_d,__fmaf_rn(p10,-x_d,p10));

   const float p = __fmaf_rn(p1,y_d,__fmaf_rn(p0,-y_d,p0));

   return p;
}

__global__ void backproject_3Dcone_beam_kernel( const float* sinogram_ptr, float* vol, const float* d_projection_matrices, const int number_of_projections,
                                                const uint3 volume_size, const float *volume_spacing_ptr, const float *volume_origin_ptr,
                                                const uint2 detector_size, 
                                                const uint3 pointer_offsets, const float *projection_multiplier)
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int k = blockIdx.z*blockDim.z + threadIdx.z;
   
   float3 volume_spacing = make_float3(*(volume_spacing_ptr+2), *(volume_spacing_ptr+1), *volume_spacing_ptr);
   float3 volume_origin = make_float3(*(volume_origin_ptr+2), *(volume_origin_ptr+1), *volume_origin_ptr);

   if( i >= volume_size.x  || j >= volume_size.y || k >= volume_size.z )
      return;
    
   const float3 coordinates = index_to_physical(make_float3(i,j,k),volume_origin,volume_spacing); 

   float val = 0.0f;
   
   for( int n = 0; n < number_of_projections; ++n )
   {
      auto ip = map(coordinates , d_projection_matrices, n );

      ip.z = 1.0f / ip.z;
      ip.x *= ip.z;
      ip.y *= ip.z;
      float3 point = make_float3(ip.x, ip.y, n);
      
      val += interp2D(sinogram_ptr, point, pointer_offsets, detector_size) * ip.z * ip.z;
   }

   // linear volume address
   const unsigned int l = volume_size.x * ( k*volume_size.y + j ) + i;
   vol[l] = (val * (*projection_multiplier));
}


void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrices,
                                          const int number_of_projections,
                                          const int volume_width, const int volume_height, const int volume_depth,
                                          const float *volume_spacing,
                                          const float *volume_origin,
                                          const int detector_width, const int detector_height, const float *projection_multiplier)
{  
   uint3 volume_size = make_uint3(volume_width, volume_height, volume_depth); 
   uint2 detector_size = make_uint2(detector_width, detector_height);
   uint3 pointer_offsets = make_uint3(1,detector_width,detector_height*detector_width);

   // launch kernel
   const unsigned int gridsize_x = (volume_size.x-1) / BLOCKSIZE_X + 1;
   const unsigned int gridsize_y = (volume_size.y-1) / BLOCKSIZE_Y + 1;
   const unsigned int gridsize_z = (volume_size.z-1) / BLOCKSIZE_Z + 1;
   const dim3 grid = dim3( gridsize_x, gridsize_y, gridsize_z );
   const dim3 block = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z );


   backproject_3Dcone_beam_kernel<<< grid, block >>>( sinogram_ptr, out, projection_matrices, number_of_projections,
                                                         volume_size, volume_spacing, volume_origin, detector_size, pointer_offsets,
                                                         projection_multiplier );


   gpuErrchk(cudaUnbindTexture(sinogram_as_texture));
}

