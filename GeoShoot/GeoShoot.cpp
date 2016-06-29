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

/* GPU MEMORY TOTAL USAGE: 24 scalar fields */
/* CPU MEMORY TOTAL USAGE: 10 * (h + 1) + 1 scalar fields  */

GeoShoot::GeoShoot(ScalarField Source, ScalarField Target, ScalarField Momentum, int N,
                   compute::command_queue queue) : Source_ { std::move(Source) },
                                                   Target_ { std::move(Target) },
                                                   Momentum_ { std::move(Momentum) },
                                                   N_ { N },
                                                   Queue_ { queue },
                                                   Convolver_ { queue } {
    assert(Source_.NX() == Momentum_.NX());
    assert(Source_.NY() == Momentum_.NY());
    assert(Source_.NZ() == Momentum_.NZ());
    assert(Source_.NX() == Target_.NX());
    assert(Source_.NY() == Target_.NY());
    assert(Source_.NZ() == Target_.NZ());

    NX_ = Source_.NX();
    NY_ = Source_.NY();
    NZ_ = Source_.NZ();

    DeltaT_ = 1.f / ((float) (N - 1));

    // fixed GPU fields
    Allocate(GPUSource_);
    Allocate(GPUInitialMomentum_);
    Allocate(GPUTarget_);
    Allocate(GPUGradientMomentum_);

    compute::copy(Source_.Begin(), Source_.End(), GPUSource_.Begin(), Queue_);
    compute::copy(Momentum_.Begin(), Momentum_.End(), GPUInitialMomentum_.Begin(), Queue_);
    compute::copy(Target_.Begin(), Target_.End(), GPUTarget_.Begin(), Queue_);

    // GPU fields used as accumulators
    Allocate(Accumulator1_);
    Allocate(Accumulator2_);
    Allocate(ScalarAccumulator1_);
    Allocate(ScalarAccumulator2_);

    // GPU fields used for computations
    Allocate(TempImageGrad_);
    Allocate(TempVelocity_);
    Allocate(TempDiffeo_);
    Allocate(TempInvDiffeo_);
    Allocate(TempImage_);
    Allocate(TempMomentum_);

    // GPU fields used for adjoint computations
    Allocate(TempAdMomentum_);
    Allocate(TempAdImage_);
    Allocate(TempAdMomentumGrad_);

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

void GeoShoot::Shoot() {
    compute::copy(
        GPUSource_.Begin(),
        GPUSource_.End(),
        TempImage_.Begin(),
        Queue_
    );
    compute::copy(
        GPUInitialMomentum_.Begin(),
        GPUInitialMomentum_.End(),
        TempMomentum_.Begin(),
        Queue_
    );

    // init diffeo = identity
    auto diffeoKernel = DiffeoKernel().create_kernel("diffeo");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    diffeoKernel.set_arg(0, TempDiffeo_.Buffer());
    diffeoKernel.set_arg(1, NX_);
    diffeoKernel.set_arg(2, NX_ * NY_);
    diffeoKernel.set_arg(3, NX_ * NY_ * NZ_);
    Queue_.enqueue_nd_range_kernel(diffeoKernel, 3, NULL, workDim, NULL);

    compute::copy(TempDiffeo_.Begin(), TempDiffeo_.End(), TempInvDiffeo_.Begin(), Queue_);

    for (auto tau = 0; tau < N_; ++tau) {
        std::cout << "Time = " << tau << std::endl;
        DiffeoTimeLine_.emplace_back(NX_, NY_, NZ_);
        compute::copy(
            TempDiffeo_.Begin(),
            TempDiffeo_.End(),
            DiffeoTimeLine_.back().Begin(),
            Queue_
        );
        
        InvDiffeoTimeLine_.emplace_back(NX_, NY_, NZ_);
        compute::copy(
            TempInvDiffeo_.Begin(),
            TempInvDiffeo_.End(),
            InvDiffeoTimeLine_.back().Begin(),
            Queue_
        );

        // transport I^0 to time tau
        if (tau != 0)
            TransportImage(GPUSource_, TempInvDiffeo_, TempImage_, Queue_);

        // transport P^0 to time tau
        if (tau != 0)
            TransportMomentum(GPUInitialMomentum_, TempInvDiffeo_, TempMomentum_, DeltaX_, Queue_);

        // compute nabla(I)
        ComputeGradScalarField(TempImage_, TempImageGrad_, DeltaX_, Queue_);

        if (tau == N_ - 1)
            break;

        // compute velocity field: v = -K (*) nabla(I)P
        ScalarFieldTimesVectorField(TempMomentum_, TempImageGrad_, TempVelocity_, -1.f, Queue_);
        Convolver_.Convolution(TempVelocity_);

        // compute phi^{tau + 1} from phi^{tau} and v^{tau}
        UpdateDiffeo(TempVelocity_, TempDiffeo_, DeltaT_, Queue_);

        // compute (phi^{-1})^{tau + 1}
        // (Accumulator1_ is used by UpdateInvDiffeo as a temporary copy of TempInvDiffeo_)
        UpdateInvDiffeo(TempVelocity_, TempInvDiffeo_, Accumulator1_, DeltaT_, DeltaX_, Queue_);
    }
}

void GeoShoot::ComputeGradient() {
    Shoot();

    // init TempAdMomentum_
    compute::fill(TempAdMomentum_.Begin(), TempAdMomentum_.End(), 0.f, Queue_);

    // init TempAdImage_
    AddFields(GPUTarget_, TempImage_, ScalarAccumulator1_, -1.f, Queue_);
    TransportMomentum(ScalarAccumulator1_, TempDiffeo_, TempAdImage_, DeltaX_, Queue_);

    for (auto k = N_ - 1; k > 0; --k) {
        if (k != N_ - 1) { // at k = N_ - 1 we have just called Shoot(), so TempDiffeo_ and TempInvDiffeo_ are up to date
            compute::copy(
                DiffeoTimeLine_.back().Begin(),
                DiffeoTimeLine_.back().End(),
                TempDiffeo_.Begin(),
                Queue_
            );

            compute::copy(
                InvDiffeoTimeLine_[k].Begin(),
                InvDiffeoTimeLine_[k].End(),
                TempInvDiffeo_.Begin(),
                Queue_
            );
        }

        TransportImage(GPUSource_, TempInvDiffeo_, TempImage_, Queue_);
        ComputeGradScalarField(TempImage_, TempImageGrad_, DeltaX_, Queue_);
        TransportMomentum(GPUInitialMomentum_, TempInvDiffeo_, TempMomentum_, DeltaX_, Queue_);

        /* compute adjoint velocity */
        ComputeGradScalarField(TempAdMomentum_, TempAdMomentumGrad_, DeltaX_, Queue_);
        ScalarFieldTimesVectorField(TempMomentum_, TempAdMomentumGrad_, Accumulator1_, 1.f, Queue_); // P x nabla(\hat{P})
        ScalarFieldTimesVectorField(TempAdImage_, TempImageGrad_, Accumulator2_, 1.f, Queue_); // \hat{I} x nabla(I)
        AddFields(Accumulator1_, Accumulator2_, TempVelocity_, -1.f, Queue_);
        Convolver_.Convolution(TempVelocity_);

        ScalarFieldTimesVectorField(TempMomentum_, TempVelocity_, Accumulator1_, 1.f, Queue_);
        // compute divergence into ScalarAccumulator1_
        TransportMomentum(ScalarAccumulator1_, TempDiffeo_, ScalarAccumulator2_, DeltaX_, Queue_);
        AddFields(TempAdImage_, ScalarAccumulator2_, TempAdImage_, DeltaT_, Queue_);

        ScalarProduct(TempImageGrad_, TempVelocity_, ScalarAccumulator1_, Queue_);
        TransportImage(ScalarAccumulator1_, TempDiffeo_, ScalarAccumulator1_, Queue_);
        AddFields(TempAdMomentum_, ScalarAccumulator1_, TempAdMomentum_, -DeltaT_, Queue_);
    }
    AddFields(TempAdMomentum_, GPUGradientMomentum_, GPUGradientMomentum_, -1.f, Queue_);
}