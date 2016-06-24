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

GeoShoot::GeoShoot(GPUScalarField Image, GPUScalarField InitialMomentum,
                   compute::command_queue queue) : Image_ { std::move(Image) },
                                                   InitialMomentum_ { std::move(InitialMomentum_) },
                                                   Queue_ { queue },
                                                   Convolver_ { queue } {

    Convolver_.InitiateConvolver(Image_.NX, Image_.NY, Image_.NZ);
}

namespace {
    compute::program & VelocityKernel() {
        static std::string source = R"#(
            __kernel void velocity(__global float * v, __global const float * gradI,
                                   __global const float * P, int NX, int NXtY, int NXtYtZ,
                                   int dir) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;
                int indDir = ind + dir * NXtYtZ; // NT == 1 so that NXtYtZtT == NXtYtZ

                v[indDir] = -P[ind] * gradI[indDir];
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void GeoShoot::Shoot(int N) {
    auto imageGrad = GPUVectorField<3>::Create(Image_.NX, Image_.NY, Image_.NZ, 1, GetContext());
    auto velocity = GPUVectorField<3>::Create(Image_.NX, Image_.NY, Image_.NZ, 1, GetContext());
    auto diffeo = GPUVectorField<3>::Create(Image_.NX, Image_.NY, Image_.NZ, 1, GetContext());
    auto image = GPUScalarField::Create(Image_.NX, Image_.NY, Image_.NZ, 1, GetContext());
    auto momentum = GPUScalarField::Create(Image_.NX, Image_.NY, Image_.NZ, 1, GetContext());

    compute::copy(Image_.field.begin(), Image_.field.end(), image.field.begin(), Queue_);
    compute::copy(
        InitialMomentum_.field.begin(),
        InitialMomentum_.field.end(),
        momentum.field.begin(),
        Queue_
    );

    float deltaT = 1. / ((float) N);
    auto velocityKernel = VelocityKernel().create_kernel("velocity");
    size_t workDim[3] = { (size_t) image.NX, (size_t) image.NY, (size_t) image.NZ };
    velocityKernel.set_arg(0, velocity.field);
    velocityKernel.set_arg(1, imageGrad.field);
    velocityKernel.set_arg(2, momentum.field);
    velocityKernel.set_arg(3, image.NX);
    velocityKernel.set_arg(4, image.NX * image.NY);
    velocityKernel.set_arg(5, image.NX * image.NY * image.NZ);

    for (auto tau = 0; tau < N; ++tau) {
        CptGradScalarField(image, imageGrad, Queue_, 0, DeltaX_);
        for (auto dir = 0; dir < 3; ++dir) {
            velocityKernel.set_arg(6, dir);
            Queue_.enqueue_nd_range_kernel(velocityKernel, 3, NULL, workDim, NULL);
        }
        Convolver_.Convolution(velocity.field, 1);

        ApplyMapping(Image_, diffeo, image, Queue_);
    }
}