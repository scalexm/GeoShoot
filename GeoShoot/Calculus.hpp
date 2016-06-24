//
//  Calculus.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef CALCULUS_HPP
#define CALCULUS_HPP

#include "GPUVectorField.hpp"

void CptGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                        compute::command_queue & queue,
                        int timeFrame = -1, float deltaX = 0.f);

void ApplyMapping(const GPUScalarField & src, const GPUVectorField<3> & mapping,
                  GPUScalarField & dst, compute::command_queue & queue, int t = 0);

#endif
