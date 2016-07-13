//
//  GeoShoot.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include "GeoShoot.hpp"
#include "GPU.hpp"
#include "Calculus.hpp"
#include "VectorField.hpp"
#include "Matrix.hpp"
#include <iostream>

/* GPU MEMORY TOTAL USAGE: 21 (GeoShoot) + 4 (FFT) scalar fields */
/* CPU MEMORY TOTAL USAGE: 2 * 3 * N scalar fields  */

GeoShoot::GeoShoot(const ScalarField & source, const ScalarField & target,
                   const ScalarField & momentum, const Matrix<4, 4> & transfo, int N,
                   compute::command_queue queue)
                   : N_ { std::max(1, N) },
                     Queue_ { queue },
                     Convolver_ { queue } {
    if (source.NX() != momentum.NX()
        || source.NY() != momentum.NY()
        || source.NZ() != momentum.NZ()
        || source.NX() != target.NX()
        || source.NY() != target.NY()
        || source.NZ() != target.NZ()) {

        throw std::invalid_argument { "images do not have the same dimensions" };
    }

    NX_ = source.NX();
    NY_ = source.NY();
    NZ_ = source.NZ();

    DeltaT_ = 1.f / ((float) (N_ - 1));

    // fixed GPU fields
    Allocate(Source_);
    Allocate(InitialMomentum_);
    Allocate(Target_);
    Allocate(GradientMomentum_);

    // GPU fields used for computations
    Allocate(Scalar1_);
    Allocate(Scalar2_);
    Allocate(Scalar3_);
    Allocate(Scalar4_);
    Allocate(Scalar5_);
    Allocate(Vector1_);
    Allocate(Vector2_);
    Allocate(Vector3_);
    Allocate(Vector4_);

    compute::copy(source.Begin(), source.End(), Source_.Begin(), Queue_);
    compute::copy(momentum.Begin(), momentum.End(), InitialMomentum_.Begin(), Queue_);
    compute::copy(target.Begin(), target.End(), Scalar1_.Begin(), Queue_);

    auto template2TargetCoord = Invert4t4Quaternion(transfo);
    template2TargetCoord = Mult4t4Quaternion(template2TargetCoord, source.Image2World());
    template2TargetCoord = Mult4t4Quaternion(target.World2Image(), template2TargetCoord);

    std::cout << "\nTemplate to target:\n";
    for (auto && r : template2TargetCoord) {
        for (auto && c : r) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }

    ProjectImage(Scalar1_, Target_, template2TargetCoord, Queue_);

    auto && mat = source.Image2World();
    Xmm_ = sqrt(mat[0][0]*mat[0][0]+mat[0][1]*mat[0][1]+mat[0][2]*mat[0][2]);
    Ymm_ = sqrt(mat[1][0]*mat[1][0]+mat[1][1]*mat[1][1]+mat[1][2]*mat[1][2]);
    Zmm_ = sqrt(mat[2][0]*mat[2][0]+mat[2][1]*mat[2][1]+mat[2][2]*mat[2][2]);

    std::cout << "Template image resolution: " << Xmm_ << " "  << Ymm_ << " "  << Zmm_ << std::endl;

    if (fabs(Xmm_ - Ymm_) > 0.0001f || fabs(Xmm_ - Zmm_) > 0.0001f) {
        std::cout << "The image-to-world matrix of the source (template) image should be isotropic"
                  << " and this is not the case here!" << std::endl;
    }

    DiffeoTimeLine_.resize(N_);
    InvDiffeoTimeLine_.resize(N_);

    for (auto n = 0; n < N_; ++n) {
        Allocate(DiffeoTimeLine_[n]);
        Allocate(InvDiffeoTimeLine_[n]);
    }
}

namespace {
    compute::program & DiffeoKernel() {
        static std::string source = R"#(
            __kernel void diffeo(__global float * diffeo,
                                   int NX, int NXtY, int NXtYtZ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;

                diffeo[ind] = x;
                diffeo[ind + NXtYtZ] = y;
                diffeo[ind + 2 * NXtYtZ] = z;
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void GeoShoot::Shooting() {
    Cost_ = Energy_ = 0.f;

    // init phi(0) = identity
    auto diffeoKernel = DiffeoKernel().create_kernel("diffeo");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    diffeoKernel.set_arg(0, Vector1_.Buffer());
    diffeoKernel.set_arg(1, NX_);
    diffeoKernel.set_arg(2, NX_ * NY_);
    diffeoKernel.set_arg(3, NX_ * NY_ * NZ_);
    Queue_.enqueue_nd_range_kernel(diffeoKernel, 3, NULL, workDim, NULL); // Vector1_ <- phi(0)

    compute::copy(Vector1_.Begin(), Vector1_.End(), Vector2_.Begin(), Queue_); // Vector2_ <- phi^{-1}(0)

    // at iteration tau: Vector1_ == phi(tau), Vector2_ == phi^{-1}(tau)
    for (auto tau = 0; tau < N_; ++tau) {
        compute::copy(
            Vector1_.Begin(),
            Vector1_.End(),
            DiffeoTimeLine_[tau].Begin(),
            Queue_
        );

        compute::copy(
            Vector2_.Begin(),
            Vector2_.End(),
            InvDiffeoTimeLine_[tau].Begin(),
            Queue_
        );

        if (tau != 0 && tau == N_ - 1) // tau = 0 needed at least for init GPUGradientMomentum_ (corner case, yes)
            break;

        TransportImage(Source_, Vector2_, Scalar1_, Queue_); // Scalar1_ <- I(tau)
        ComputeGradScalarField(Scalar1_, Vector4_, DeltaX_, Queue_); // Vector4_ <- nabla(I)

        TransportMomentum(InitialMomentum_, Vector2_, Scalar1_, DeltaX_, Queue_); // Scalar1_ <- P(tau)

        ScalarFieldTimesVectorField(Scalar1_, Vector4_, Vector3_, -1.f, Queue_); // Vector3_ <- -P(tau) x nabla(I)
        Convolver_.Convolution(Vector3_); // Vector3_ <- -K (*) [P(tau) x nabla(I)] = v(tau)

        if (tau == 0) { // init GradientMomentum_
            ScalarProduct(Vector4_, Vector3_, GradientMomentum_, -Alpha, Queue_);
            Energy_ +=
                0.5 * DotProduct(GradientMomentum_, InitialMomentum_, Scalar1_, Queue_);
            std::cout << "Energy of the vector field: " << Energy_ << "\n";

            if (N_ == 1)
                break;
        }

        UpdateDiffeo(Vector3_, Vector1_, DeltaT_, Queue_); // Vector1_ <- phi(tau + 1)

        // (Vector4_ is used by UpdateInvDiffeo as a temporary copy of Vector2_)
        UpdateInvDiffeo(Vector3_, Vector2_, Vector4_, DeltaT_, DeltaX_, Queue_); // Vector2_ <- phi(tau + 1)
    }
}

void GeoShoot::ComputeGradient() {
    TransportImage(Source_, Vector2_, Scalar3_, Queue_); // Scalar3_ <- I(1) (Vector2_ still contains phi^{-1}(1))

    using compute::lambda::get;
    using compute::_1;

    compute::for_each(
        compute::make_zip_iterator(
            boost::make_tuple(Scalar3_.Begin(), Target_.Begin(), Scalar1_.Begin())
        ),
        compute::make_zip_iterator(
            boost::make_tuple(Scalar3_.End(), Target_.End(), Scalar1_.End())
        ),
        get<2>(_1) = pow(get<0>(_1) - get<1>(_1), 2),
        Queue_
    );

    auto similarityMeasure = 0.f;
    compute::reduce(
        Scalar1_.Begin(),
        Scalar1_.End(),
        &similarityMeasure,
        compute::plus<float>(),
        Queue_
    );

    Cost_ += similarityMeasure * 0.5f;
    std::cout << "Similarity measure: " << Cost_ << "\n";

    Cost_ += Energy_;
    std::cout << "Global cost: " << Cost_ << "\n";

    AddFields(Target_, Scalar3_, Scalar4_, -1.f, Queue_); // Scalar4_ <- J - I(1) = \hat{I}(1)
    TransportMomentum(Scalar4_, Vector1_, Scalar1_, DeltaX_, Queue_); // Scalar1_ <- \tilde{I}(1) (Vector1_ still conntains phi(1))

    Scalar2_.Fill(0.f, Queue_); // Scalar2_ <- \tilde{P}(1) = 0

    // at iteration tau: Scalar1_ == \tilde{I}(tau), Scalar2_ == \tilde{P}(tau)
    for (auto tau = N_ - 1; tau > 0; --tau) {
        if (tau != N_ - 1) {
            // at tau = N_ - 1 we have just called Shoot(), so Vector1_ still contains phi(1)
            compute::copy(
                DiffeoTimeLine_[tau].Begin(),
                DiffeoTimeLine_[tau].End(),
                Vector1_.Begin(),
                Queue_
            );

            // same: Vector2_ still contains phi^{-1}(1)
            compute::copy(
                InvDiffeoTimeLine_[tau].Begin(),
                InvDiffeoTimeLine_[tau].End(),
                Vector2_.Begin(),
                Queue_
            );

            // at tau = N_ - 1, Scalar3_ already contains I(1)
            TransportImage(Source_, Vector2_, Scalar3_, Queue_); // Scalar3_ <- I(tau)

            // at tau = N - 1, Scalar4_ already contains \hat{I}(1)
            TransportMomentum(Scalar1_, Vector2_, Scalar4_, DeltaX_, Queue_); // Scalar4_ <- \hat{I}(tau)
        }

        TransportMomentum(Scalar2_, Vector2_, Scalar5_, DeltaX_, Queue_); // Scalar5_ <- \hat{P}(tau)

        ComputeGradScalarField(Scalar3_, Vector3_, DeltaX_, Queue_); // Vector3_ <- nabla(I)

        TransportMomentum(InitialMomentum_, Vector2_, Scalar3_, DeltaX_, Queue_); // Scalar3_ <- P(tau)

        /* compute adjoint velocity */
        ComputeGradScalarField(Scalar5_, Vector2_, DeltaX_, Queue_); // Vector2_ <- nabla(\hat{P}) (we don't use phi^{-1} anymore so we can overwrite Vector2_)
        ScalarFieldTimesVectorField(Scalar3_, Vector2_, Vector2_, 1.f, Queue_); // Vector2_ <- P x nabla(\hat{P})
        ScalarFieldTimesVectorField(Scalar4_, Vector3_, Vector4_, 1.f, Queue_); // Vector4_ <- \hat{I} x nabla(I)
        AddFields(Vector2_, Vector4_, Vector2_, -1.f, Queue_); // Vector2_ <- P x nabla(\hat{P}) - \hat{I} x nabla(I)
        Convolver_.Convolution(Vector2_); // Vector2_ <- \hat{v}

        ScalarFieldTimesVectorField(Scalar3_, Vector2_, Vector4_, 1.f, Queue_); // Vector4_ <- P x \hat{v}
        ComputeDivVectorField(Vector4_, Scalar4_, DeltaX_, Queue_); // Scalar4_ <- div(P x \hat{v})
        TransportMomentum(Scalar4_, Vector1_, Scalar3_, DeltaX_, Queue_); // Scalar3_ <- Jac(phi) div(P x \hat{v}) o phi
        AddFields(Scalar1_, Scalar3_, Scalar1_, DeltaT_, Queue_); // Scalar1_ <- \tilde{I}(tau - 1)

        ScalarProduct(Vector3_, Vector2_, Scalar3_, 1.f, Queue_); // Scalar3_ <- nabla(I) . \hat{v}
        TransportImage(Scalar3_, Vector1_, Scalar4_, Queue_); // Scalar4_ <- [nabla(I) . \hat{v}] o phi
        AddFields(Scalar2_, Scalar4_, Scalar2_, -DeltaT_, Queue_); // Scalar2_ <- \tilde{P}(tau - 1)
    }

    AddFields(GradientMomentum_, Scalar2_, GradientMomentum_, -1.f, Queue_); // update GradientMomentum_
}

void GeoShoot::GradientDescent(int iterationsNumber, float gradientStep) {
    auto optimizedCost = Source_.MaxAbsVal(Queue_) + Target_.MaxAbsVal(Queue_);
    optimizedCost *= NX_ * NY_ * NZ_;

    auto currentCost = optimizedCost;
    auto localCounter = 0;
    for (auto i = 0; i < iterationsNumber; ++i) {
        std::cout << std::endl;
        std::cout << "\tGradient iteration number " << (i + 1) << "\n";
        Shooting();
        ComputeGradient();

        if (Cost_ < optimizedCost) {
            optimizedCost = Cost_;
            localCounter = 0;
        }

        if (Cost_ > currentCost) {
            ++localCounter;
            if (localCounter == 2) {
                gradientStep *= 0.8f;
                localCounter = 0;
            }
        }

        currentCost = Cost_;
        ComputeGradScalarField(Source_, Vector1_, DeltaX_, Queue_);
        ScalarFieldTimesVectorField(GradientMomentum_, Vector1_, Vector1_, -1.f, Queue_);
        Convolver_.Convolution(Vector1_);

        auto temp = MaxUpdate / Vector1_.MaxAbsVal(Queue_);

        AddFields(
            InitialMomentum_,
            GradientMomentum_,
            InitialMomentum_,
            -temp * gradientStep,
            Queue_
        );

        /*if (i == 20) {
            auto m = ScalarField { NX_, NY_, NZ_ };
            compute::copy(InitialMomentum_.Begin(), InitialMomentum_.End(), m.Begin(), Queue_);
            m.Write({"/Users/alexm/Desktop/lol2.nii"});
            exit(0);
        }*/
    }
}

void GeoShoot::Run(int iterationsNumber) {
    iterationsNumber = std::max(1, iterationsNumber);

    if (!Init_) {
        for (auto i = 0; i < 7; ++i) {
            SigmaXs[i] /= Xmm_;
            SigmaYs[i] /= Ymm_;
            SigmaZs[i] /= Zmm_;
        }

        if (fabs(Weights[0]) > 0.01f) {
            auto sum = std::accumulate(Weights.begin(), Weights.end(), 0.f);
            for (auto && w : Weights)
                w /= sum;
        }

        Convolver_.InitiateConvolver(
            NX_, NY_, NZ_,
            Weights,
            SigmaXs,
            SigmaYs,
            SigmaZs
        );

        if (fabs(Weights[0]) < 0.01f)
            ReInitiateConvolver_HomoAppaWeights();

        Init_ = true;
    }

    GradientDescent(iterationsNumber, 0.5f);
}

void GeoShoot::Save(std::string path) {
    if (path.back() != '/')
        path += '/';

    auto momentum = ScalarField { NX_, NY_, NZ_ };
    compute::copy(InitialMomentum_.Begin(), InitialMomentum_.End(), momentum.Begin(), Queue_);
    auto momentumPath = path + "Momentum.nii";
    momentum.Write({ momentumPath.c_str() });

    auto final = ScalarField { NX_, NY_, NZ_ };
    auto finalPath = path + "FinalDefSrc.nii";
    compute::copy(
        InvDiffeoTimeLine_.back().Begin(),
        InvDiffeoTimeLine_.back().End(),
        Vector1_.Begin(),
        Queue_
    );
    TransportImage(Source_, Vector1_, Scalar1_, Queue_);
    compute::copy(Scalar1_.Begin(), Scalar1_.End(), final.Begin(), Queue_);
    final.Write({ finalPath.c_str() });
}

namespace {
    compute::program & MaxGradKernel() {
        static std::string source = R"#(
            __kernel void maxGrad(__global float * dst, __global const float * field,
                                  int NX, int NY, int NZ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int ind = x + y * NX + z * NXtY;

                dst[ind] = 2.f * sqrt(field[ind] * field[ind]
                    + field[ind + NXtYtZ] * field[ind + NXtYtZ]
                    + field[ind + 2 * NXtYtZ] * field[ind + 2 * NXtYtZ]);
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void GeoShoot::ReInitiateConvolver_HomoAppaWeights() {
    ComputeGradScalarField(Target_, Vector1_, DeltaX_, Queue_);

    float realW1;

    auto kernel = MaxGradKernel().create_kernel("maxGrad");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    kernel.set_arg(0, Scalar1_.Buffer());
    kernel.set_arg(1, Vector2_.Buffer());
    kernel.set_arg(2, NX_);
    kernel.set_arg(3, NY_);
    kernel.set_arg(4, NZ_);

    for (auto i = 0; i < 7; ++i) {
        Convolver_.ChangeKernel(
            { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { SigmaXs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
            { SigmaYs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
            { SigmaZs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f }
        );

        AddFields(Source_, Target_, Scalar1_, -1.f, Queue_);
        ScalarFieldTimesVectorField(Scalar1_, Vector1_, Vector2_, 1.f, Queue_);

        Convolver_.Convolution(Vector2_);

        Queue_.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
        auto it = compute::max_element(Scalar1_.Begin(), Scalar1_.End(), Queue_);
        auto maxGrad = it.read(Queue_);

        if (i == 0) {
            Weights[i] = 100.f;
            realW1 = 1.f / maxGrad;
        } else
            Weights[i] = 100.f / maxGrad / realW1;

        std::cout << "sigma" << (i + 1) << " = " << (SigmaXs[i] * Xmm_)
                  << " / weight" << (i + 1) << " = " << Weights[i] << "\n";
    }

    std::cout << std::endl;
    Convolver_.ChangeKernel(Weights, SigmaXs, SigmaYs, SigmaZs);
}