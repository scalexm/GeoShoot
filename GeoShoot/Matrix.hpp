/*=========================================================================
 
 Matrix.hpp
 GeoShoot

 Author: Alexandre Martin, Laurent Risser
 Copyright © 2016 scalexm, lrisser. All rights reserved.
 
 Disclaimer: This software has been developed for research purposes only, and hence should 
 not be used as a diagnostic tool. In no event shall the authors or distributors
 be liable to any direct, indirect, special, incidental, or consequential 
 damages arising of the use of this software, its documentation, or any 
 derivatives thereof, even if the authors have been advised of the possibility 
 of such damage. 

 =========================================================================*/

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

Matrix<4, 4> Invert4t4Quaternion(const Matrix<4, 4> &);
Matrix<4, 4> Mult4t4Quaternion(const Matrix<4, 4> &, const Matrix<4, 4> &);

#endif
