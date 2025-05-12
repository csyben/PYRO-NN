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
 * Computes line intersection for projector kernel
 * Implementation is adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#pragma once
#ifndef HELPER_GEOMETRY_GPU_H
#define HELPER_GEOMETRY_GPU_H

#include "helper_math.h"

inline __device__ float2 intersectLines2D(float2 p1, float2 p2, float2 p3, float2 p4)
{
    float dNom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

    if (dNom < 0.000001f && dNom > -0.000001f)
    {
        float2 retValue = {NAN, NAN};
        return retValue;
    }
    float x = (p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x);
    float y = (p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x);

    x /= dNom;
    y /= dNom;
    float2 isectPt = {x, y};
    return isectPt;
}
#endif