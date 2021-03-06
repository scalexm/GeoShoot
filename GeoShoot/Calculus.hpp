/*=========================================================================
 
 Calculus.hpp
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

#ifndef CALCULUS_HPP
#define CALCULUS_HPP

#include "VectorField.hpp"

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

void ComputeGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                            compute::command_queue & queue);

void ComputeDivVectorField(const GPUVectorField<3> & field, GPUScalarField & div,
                           compute::command_queue & queue);

// cannot be used in place
void TransportImage(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                    GPUScalarField & dst, compute::command_queue & queue);

// cannot be used in place
void TransportMomentum(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                       GPUScalarField & dst, compute::command_queue & queue);

void UpdateDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                  compute::command_queue & queue);

// needs an accumulator
void UpdateInvDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo,
                     GPUVectorField<3> & accumulator, float deltaT,
                     compute::command_queue & queue);

// can be used in place
template<class Field>
void AddFields(const Field & src1, const Field & src2, Field & dst,
               float factor, compute::command_queue & queue) {
    assert(src1.NX() == src2.NX());
    assert(src1.NY() == src2.NY());
    assert(src1.NZ() == src2.NZ());
    assert(src1.NX() == dst.NX());
    assert(src1.NY() == dst.NY());
    assert(src1.NZ() == dst.NZ());

    using compute::lambda::get;
    using compute::_1;

    compute::for_each(
        compute::make_zip_iterator(
            boost::make_tuple(src1.Begin(), src2.Begin(), dst.Begin())
        ),
        compute::make_zip_iterator(
            boost::make_tuple(src1.End(), src2.End(), dst.End())
        ),
        get<2>(_1) = get<0>(_1) + factor * get<1>(_1),
        queue
    );
}

// can be used in place
void ScalarFieldTimesVectorField(const GPUScalarField & src1, const GPUVectorField<3> & src2,
                                 GPUVectorField<3> & dst, float factor,
                                 compute::command_queue & queue);

void ScalarProduct(const GPUVectorField<3> & src1, const GPUVectorField<3> & src2,
                   GPUScalarField & dst, float factor, compute::command_queue & queue);

float DotProduct(const GPUScalarField & src1, const GPUScalarField & src2,
                 GPUScalarField & accumulator, compute::command_queue & queue);

void ProjectImage(const GPUScalarField & src, GPUScalarField & dst, const Matrix<4, 4> & transfo,
                  compute::command_queue & queue);

#endif
