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

GeoShoot::GeoShoot(ScalarField Image, ScalarField Momentum, int N,
                   compute::command_queue queue) : Image_ { std::move(Image) },
                                                   Momentum_ { std::move(Momentum) },
                                                   N_ { N },
                                                   Queue_ { queue },
                                                   Convolver_ { queue } {
    assert(Image_.NX() == Momentum_.NX());
    assert(Image_.NY() == Momentum_.NY());
    assert(Image_.NZ() == Momentum_.NZ());

    NX_ = Image_.NX();
    NY_ = Image_.NY();
    NZ_ = Image_.NZ();

    DeltaT_ = 1.f / ((float) (N - 1));

    TempImageGrad_ = GPUVectorField<3> { NX_, NY_, NZ_, GetContext() };
    TempVelocity_ = GPUVectorField<3> { NX_, NY_, NZ_, GetContext() },
    TempDiffeo_ = GPUVectorField<3> { NX_, NY_, NZ_, GetContext() },
    TempInvDiffeo_ = GPUVectorField<3> { NX_, NY_, NZ_, GetContext() };
    TempInitialImage_ = GPUScalarField { NX_, NY_, NZ_, GetContext() },
    TempImage_ = GPUScalarField { NX_, NY_, NZ_, GetContext() },
    TempInitialMomentum_ = GPUScalarField { NX_, NY_, NZ_, GetContext() },
    TempMomentum_ = GPUScalarField { NX_, NY_, NZ_, GetContext() };

    compute::copy(Image_.Begin(), Image_.End(), TempInitialImage_.Begin(), Queue_);
    compute::copy(Momentum_.Begin(), Momentum_.End(), TempInitialMomentum_.Begin(), Queue_);

    Convolver_.InitiateConvolver(Image_.NX(), Image_.NY(), Image_.NZ(),
        { 100.f, 80.3515f, 63.4543f, 49.7454f, 38.8999f, 30.5439f, 24.2221f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f },
        { 2.5f, 2.25f, 2.1f, 1.75f, 1.5f, 1.25f, 1.f });
}

namespace {
    compute::program & VelocityKernel() {
        static std::string source = R"#(
            __kernel void velocity(__global float * v, __global const float * nablaI,
                                   __global const float * P, int NX, int NXtY, int NXtYtZ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;

                v[ind] = -P[ind] * nablaI[ind];
                v[ind + NXtYtZ] = -P[ind] * nablaI[ind + NXtYtZ];
                v[ind + 2 * NXtYtZ] = -P[ind] * nablaI[ind + 2 * NXtYtZ];
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

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
        TempInitialImage_.Begin(),
        TempInitialImage_.End(),
        TempImage_.Begin(),
        Queue_
    );
    compute::copy(
        TempInitialMomentum_.Begin(),
        TempInitialMomentum_.End(),
        TempMomentum_.Begin(),
        Queue_
    );

    auto velocityKernel = VelocityKernel().create_kernel("velocity");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    velocityKernel.set_arg(0, TempVelocity_.Buffer());
    velocityKernel.set_arg(1, TempImageGrad_.Buffer());
    velocityKernel.set_arg(2, TempMomentum_.Buffer());
    velocityKernel.set_arg(3, NX_);
    velocityKernel.set_arg(4, NX_ * NY_);
    velocityKernel.set_arg(5, NX_ * NY_ * NZ_);

    // init diffeo = identity
    auto diffeoKernel = DiffeoKernel().create_kernel("diffeo");
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
            TransportImage(TempInitialImage_, TempInvDiffeo_, TempImage_, Queue_);

        // transport P^0 to time tau
        if (tau != 0)
            TransportMomentum(TempInitialMomentum_, TempInvDiffeo_, TempMomentum_, DeltaX_, Queue_);

        // compute nabla(I)
        CptGradScalarField(TempImage_, TempImageGrad_, DeltaX_, Queue_);

        if (tau == N_ - 1)
            break;

        // compute velocity field: v = -K (*) nabla(I)P
        Queue_.enqueue_nd_range_kernel(velocityKernel, 3, NULL, workDim, NULL);
        Convolver_.Convolution(TempVelocity_);

        // compute phi^{tau + 1} from phi^{tau} and v^{tau}
        UpdateDiffeo(TempVelocity_, TempDiffeo_, DeltaT_, Queue_);

        // compute (phi^{-1})^{tau + 1}
        UpdateInvDiffeo(TempVelocity_, TempInvDiffeo_, DeltaT_, DeltaX_, Queue_);
    }
}

void GeoShoot::ComputeGradient() {
    
}