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
/* CPU MEMORY TOTAL USAGE: 10 * (h + 1) + 1 scalar fields  */

GeoShoot::GeoShoot(const ScalarField & Source, const ScalarField & Target,
                   const ScalarField & Momentum, int N, compute::command_queue queue)
                   : N_ { N },
                     Queue_ { queue },
                     Convolver_ { queue } {
    assert(Source.NX() == Momentum.NX());
    assert(Source.NY() == Momentum.NY());
    assert(Source.NZ() == Momentum.NZ());
    assert(Source.NX() == Target.NX());
    assert(Source.NY() == Target.NY());
    assert(Source.NZ() == Target.NZ());

    NX_ = Source.NX();
    NY_ = Source.NY();
    NZ_ = Source.NZ();

    DeltaT_ = 1.f / ((float) (N_ - 1));

    // fixed GPU fields
    Allocate(Source_);
    Allocate(InitialMomentum_);
    Allocate(Target_);
    Allocate(GradientMomentum_);

    compute::copy(Source.Begin(), Source.End(), Source_.Begin(), Queue_);
    compute::copy(Momentum.Begin(), Momentum.End(), InitialMomentum_.Begin(), Queue_);
    compute::copy(Target.Begin(), Target.End(), Target_.Begin(), Queue_);

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

    Convolver_.InitiateConvolver(NX_, NY_, NZ_,
        { 100.f, 80.3515f, 63.4543f, 49.7454f, 38.8999f, 30.5439f, 24.2221f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f });
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
        DiffeoTimeLine_.emplace_back(NX_, NY_, NZ_);
        compute::copy(
            Vector1_.Begin(),
            Vector1_.End(),
            DiffeoTimeLine_.back().Begin(),
            Queue_
        );
        
        InvDiffeoTimeLine_.emplace_back(NX_, NY_, NZ_);
        compute::copy(
            Vector2_.Begin(),
            Vector2_.End(),
            InvDiffeoTimeLine_.back().Begin(),
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
            ScalarProduct(Vector4_, Vector3_, GradientMomentum_, -Alpha_, Queue_);
            Energy_ +=
                0.5 * DotProduct(GradientMomentum_, InitialMomentum_, Scalar1_, Queue_);
            std::cout << "Energy of the vector field: " << Energy_ << std::endl;

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
    std::cout << "Similarity measure: " << Cost_ << std::endl;

    Cost_ += Energy_;
    std::cout << "Global cost: " << Cost_ << std::endl;

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
        DiffeoTimeLine_.clear();
        InvDiffeoTimeLine_.clear();
        std::cout << "\tGradient iteration number " << (i + 1) << std::endl;
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

        auto temp = MaxUpdate_ / Vector1_.MaxAbsVal(Queue_);

        AddFields(
            InitialMomentum_,
            GradientMomentum_,
            InitialMomentum_,
            -temp * gradientStep, Queue_
        );
    }
}

void GeoShoot::Run(int iterationsNumber) {
    GradientDescent(iterationsNumber, 0.5f);

    auto m = ScalarField { NX_, NY_, NZ_ };
    compute::copy(InitialMomentum_.Begin(), InitialMomentum_.End(), m.Begin(), Queue_);
    m.Write({"/Users/alexm/Desktop/m.nii"});
}