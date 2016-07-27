/*=========================================================================
 
 GeoShoot.hpp
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

#ifndef GEO_SHOOT_HPP
#define GEO_SHOOT_HPP

#include "FFTConvolver.hpp"

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

class GeoShoot {
private:
    VectorField<3> DiffeoTimeLine_, InvDiffeoTimeLine_;

    compute::command_queue Queue_;
    FFTConvolver Convolver_;
    float DeltaT_, Cost_, Energy_, Xmm_, Ymm_, Zmm_;

    int N_;
    int NX_, NY_, NZ_;

    bool Init_ = false;

    GPUScalarField Source_, Target_, InitialMomentum_, GradientMomentum_;

    GPUScalarField Scalar1_, Scalar2_, Scalar3_, Scalar4_, Scalar5_;
    GPUVectorField<3> Vector1_, Vector2_, Vector3_, Vector4_;

    template<class T>
    void Allocate(T & field) {
        field = T { NX_, NY_, NZ_, 1, Queue_.get_context() };
    }

    template<class T>
    void Deallocate(T & field) {
        field = T { };
    }

    compute::int4_ Dims() const {
        return { NX_, NY_, NZ_, 0 };
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

    std::array<float, 7> Weights = {{ 100.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f }},
        SigmaXs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }},
        SigmaYs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }},
        SigmaZs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }};

    float Alpha = 0.001f, MaxUpdate = 0.5f;
};

#endif
