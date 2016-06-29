//
//  GeoShoot.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef GEO_SHOOT_HPP
#define GEO_SHOOT_HPP

#include "FFTConvolver.hpp"

class GeoShoot {
private:
    std::vector<VectorField<3>> DiffeoTimeLine_, InvDiffeoTimeLine_;

    ScalarField Source_, Target_, Momentum_;
    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaX_ = 1.f, DeltaT_;

    int N_;
    int NX_, NY_, NZ_;

    GPUScalarField GPUSource_, GPUTarget_, GPUInitialMomentum_, GPUGradientMomentum_;

    GPUVectorField<3> Accumulator1_, Accumulator2_;
    GPUScalarField ScalarAccumulator1_, ScalarAccumulator2_;

    GPUVectorField<3> TempImageGrad_, TempVelocity_, TempDiffeo_, TempInvDiffeo_;
    GPUScalarField TempImage_, TempMomentum_;

    GPUVectorField<3> TempAdMomentumGrad_;
    GPUScalarField TempAdMomentum_, TempAdImage_;

    template<class T>
    void Allocate(T & field) {
        field = T { NX_, NY_, NZ_, Queue_.get_context() };
    }

    template<class T>
    void Deallocate(T & field) {
        field = T { };
    }
public:
    GeoShoot(ScalarField Source, ScalarField Target, ScalarField Momentum, int N,
             compute::command_queue queue);
    void Shoot();
    void ComputeGradient();
};

#endif
