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

    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaX_ = 1.f, DeltaT_, Cost_, Energy_, Alpha_ = 0.001f, MaxUpdate_ = 0.5f;

    int N_;
    int NX_, NY_, NZ_;

    GPUScalarField Source_, Target_, InitialMomentum_, GradientMomentum_;

    GPUScalarField Scalar1_, Scalar2_, Scalar3_, Scalar4_, Scalar5_;
    GPUVectorField<3> Vector1_, Vector2_, Vector3_, Vector4_;

    template<class T>
    void Allocate(T & field) {
        field = T { NX_, NY_, NZ_, Queue_.get_context() };
    }

    template<class T>
    void Deallocate(T & field) {
        field = T { };
    }

    void Shooting();
    void ComputeGradient();
    void GradientDescent(int, float);
public:
    GeoShoot(const ScalarField & Source, const ScalarField & Target, const ScalarField & Momentum,
             int N, compute::command_queue queue);

    void Run(int iterationsNumber);
};

#endif
