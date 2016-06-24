//
//  GeoShoot.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef GEO_SHOOT_HPP
#define GEO_SHOOT_HPP

#include "GPUVectorField.hpp"
#include "FFTConvolver.hpp"

class GeoShoot {
private:
    GPUScalarField Image_, InitialMomentum_;
    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaX_ = 1.f;

public:
    GeoShoot(GPUScalarField Image, GPUScalarField InitialMomentum, compute::command_queue queue);
    void Shoot(int N);
};

#endif
