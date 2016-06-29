//
//  Calculus.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef CALCULUS_HPP
#define CALCULUS_HPP

#include "VectorField.hpp"

void CptGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                        float deltaX, compute::command_queue & queue);

void TransportImage(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                    GPUScalarField & dst, compute::command_queue & queue);

void TransportMomentum(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                       GPUScalarField & dst, float deltaX, compute::command_queue & queue);

void UpdateDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                  compute::command_queue & queue);

void UpdateInvDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                     float deltaX, compute::command_queue & queue);

#endif
