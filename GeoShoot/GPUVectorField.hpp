//
//  GPUVectorField.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef GPU_VECTOR_FIELD_HPP
#define GPU_VECTOR_FIELD_HPP

#include <boost/compute.hpp>

namespace compute = boost::compute;

template<size_t Dim>
struct GPUVectorField {
    compute::vector<float> field;
    int NX, NY, NZ, NT;

    GPUVectorField() = default;

    GPUVectorField(int NX, int NY, int NZ, int NT, const compute::context & ctx)
        : NX { NX }, NY { NY }, NZ { NZ }, NT { NT }, field(Dim * NX * NY * NZ * NT, ctx) {
    }

    GPUVectorField(const GPUVectorField &) = delete;
    GPUVectorField(GPUVectorField &&) = default;
    GPUVectorField & operator =(const GPUVectorField &) = delete;
    GPUVectorField & operator =(GPUVectorField &&) = default;
};

using GPUScalarField = GPUVectorField<1>;

#endif
