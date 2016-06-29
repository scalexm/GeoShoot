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

    ScalarField Image_, Momentum_;
    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaX_ = 1.f, DeltaT_;

    int N_;
    int NX_, NY_, NZ_;

    GPUVectorField<3> TempImageGrad_, TempVelocity_, TempDiffeo_, TempInvDiffeo_;
    GPUScalarField TempInitialImage_, TempImage_, TempInitialMomentum_, TempMomentum_;
public:
    GeoShoot(ScalarField Image, ScalarField Momentum, int N, compute::command_queue queue);
    void Shoot();
    void ComputeGradient();
};

#endif
