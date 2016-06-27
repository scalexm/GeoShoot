//
//  Matrix.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

Matrix<4, 4> Invert4t4Quaternion(const Matrix<4, 4> &);

void TridiagonalSolveFloat(const float * a, const float * b, float * c, float * d,
                           float *x, int n);

#endif
