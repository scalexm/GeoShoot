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

    static GPUVectorField Create(int NX, int NY, int NZ, int NT, const compute::context & ctx) {
        GPUVectorField<Dim> field;
        field.NX = NX;
        field.NY = NY;
        field.NZ = NZ;
        field.NT = NT;
        field.field = compute::vector<float>(Dim * NX * NY * NZ * NT, ctx);
        return field;
    }

    GPUVectorField() = default;
    GPUVectorField(const GPUVectorField &) = delete;
    GPUVectorField(GPUVectorField &&) = default;
    GPUVectorField & operator =(const GPUVectorField &) = delete;
    GPUVectorField & operator =(GPUVectorField &&) = default;
};

using GPUScalarField = GPUVectorField<1>;

#endif
