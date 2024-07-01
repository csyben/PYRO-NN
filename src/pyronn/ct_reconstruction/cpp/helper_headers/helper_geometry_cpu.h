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
 * Helper methods to prepare projection matrices for projector kernel
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#ifndef HELPER_GEOMETRY_CPU_H
#define HELPER_GEOMETRY_CPU_H
#pragma once

#include <math_constants.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace Geometry{
    
    /// Compute right null-space of A
    Eigen::VectorXf nullspace(const Eigen::MatrixXf& A)
    {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto V=svd.matrixV();
        return V.col(V.cols()-1);
    }
    /// Extract the world coordinates of the camera center from a projection matrix. (SVD based implementation)
    Eigen::Vector4f getCameraCenter(const Eigen::MatrixXf& P)
    {
        Eigen::Vector4f C = Geometry::nullspace(P);
        if (C(3)<-1e-12 || C(3)>1e-12)
            C=C/C(3); // Def:Camera centers are always positive.
        return C;
    }
}
#endif