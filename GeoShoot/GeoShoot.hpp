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

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

class GeoShoot {
private:
    std::vector<VectorField<3>> DiffeoTimeLine_, InvDiffeoTimeLine_;

    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaX_ = 1.f, DeltaT_, Cost_, Energy_, Xmm_, Ymm_, Zmm_;

    int N_;
    int NX_, NY_, NZ_;

    bool Init_ = false;

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

    void ReInitiateConvolver_HomoAppaWeights();

    void Shooting();
    void ComputeGradient();
    void GradientDescent(int, float);
public:
    GeoShoot(const ScalarField & source, const ScalarField & target, const ScalarField & momentum,
            const Matrix<4, 4> & transfo, int N, compute::command_queue queue);

    void Run(int iterationsNumber);
    void Save(std::string path);

    std::array<float, 7> Weights = { 100.f, 80.3515f, 63.4543f, 49.7454f, 38.8999f, 30.5439f, 24.2221f },
        SigmaXs = { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        SigmaYs = { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        SigmaZs = { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f };

    float Alpha = 0.001f, MaxUpdate = 0.5f;
};

#endif
