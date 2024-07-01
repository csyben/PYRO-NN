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
 * Helper methods to convert index to physical coordinates and vice versa
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#ifndef HELPER_GRID_H
#define HELPER_GRID_H


inline __host__ __device__ float index_to_physical(float index, float origin, float spacing)
{
    return index * spacing + origin;
}

inline __host__ __device__ float physical_to_index(float physical, float origin, float spacing)
{
    return (physical - origin) / spacing;
}

inline __host__ __device__ float2 index_to_physical(float2 index, float2 origin, float2 spacing)
{
    return make_float2(index.x * spacing.x + origin.x, index.y * spacing.y + origin.y);
}

inline __host__ __device__ float2 physical_to_index(float2 physical, float2 origin, float2 spacing)
{
    return make_float2((physical.x - origin.x) / spacing.x, (physical.y - origin.y) / spacing.y);
}

inline __host__ __device__ float3 index_to_physical(float3 index, float3 origin, float3 spacing)
{
    return make_float3(index.x * spacing.x + origin.x, index.y * spacing.y + origin.y, index.z * spacing.z + origin.z);
}

inline __host__ __device__ float3 physical_to_index(float3 physical, float3 origin, float3 spacing)
{
    return make_float3((physical.x - origin.x) / spacing.x, (physical.y - origin.y) / spacing.y, (physical.z - origin.z) / spacing.z);
}

#endif


