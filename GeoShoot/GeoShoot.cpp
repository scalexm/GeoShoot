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

GeoShoot::GeoShoot(GPUScalarField Image, GPUScalarField InitialMomentum,
                   compute::command_queue queue) : Image_ { std::move(Image) },
                                                   InitialMomentum_ { std::move(InitialMomentum) },
                                                   Queue_ { queue },
                                                   Convolver_ { queue } {

    assert(Image_.NX == InitialMomentum_.NX);
    assert(Image_.NY == InitialMomentum_.NY);
    assert(Image_.NZ == InitialMomentum_.NZ);
    Convolver_.InitiateConvolver(Image_.NX, Image_.NY, Image_.NZ);
}

namespace {
    compute::program & VelocityKernel() {
        static std::string source = R"#(
            __kernel void velocity(__global float * v, __global const float * nablaI,
                                   __global const float * P, int NX, int NXtY, int NXtYtZtT,
                                   int dir) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;
                int indDir = ind + dir * NXtYtZtT;

                v[indDir] = -P[ind] * nablaI[indDir];
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program & DiffeoKernel() {
        static std::string source = R"#(
            __kernel void diffeo(__global float * diffeo,
                                   int NX, int NXtY, int NXtYtZtT) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;

                diffeo[ind] = x;
                diffeo[ind + NXtYtZtT] = y;
                diffeo[ind + 2 * NXtYtZtT] = z;
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program & UpdateDiffeoKernel() {
        static std::string source = R"#(
            __kernel void updateDiffeo(__global float * diffeo, __global const float * velocity,
                                   int NX, int NXtY, int NXtYtZtT, float deltaT) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;

                diffeo[ind] += velocity[ind] * deltaT;
                diffeo[ind + NXtYtZtT] += velocity[ind + NXtYtZtT] * deltaT;
                diffeo[ind + 2 * NXtYtZtT] += velocity[ind + 2 * NXtYtZtT] * deltaT;
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void GeoShoot::Shoot(int N) {
    auto NX = Image_.NX, NY = Image_.NY, NZ = Image_.NZ;
    auto imageGrad = GPUVectorField<3> { NX, NY, NZ, 1, GetContext() };
    auto velocity = GPUVectorField<3> { NX, NY, NZ, 1, GetContext() };
    auto diffeo = GPUVectorField<3> { NX, NY, NZ, 1, GetContext() };
    auto image = GPUScalarField { NX, NY, NZ, 1, GetContext() };
    auto momentum = GPUScalarField { NX, NY, NZ, 1, GetContext() };

    compute::copy(Image_.field.begin(), Image_.field.end(), image.field.begin(), Queue_);
    compute::copy(
        InitialMomentum_.field.begin(),
        InitialMomentum_.field.end(),
        momentum.field.begin(),
        Queue_
    );

    auto deltaT = 1.f / ((float) N);

    auto velocityKernel = VelocityKernel().create_kernel("velocity");
    size_t workDim[3] = { (size_t) NX, (size_t) NY, (size_t) NZ };
    velocityKernel.set_arg(0, velocity.field);
    velocityKernel.set_arg(1, imageGrad.field);
    velocityKernel.set_arg(2, momentum.field);
    velocityKernel.set_arg(3, NX);
    velocityKernel.set_arg(4, NX * NY);
    velocityKernel.set_arg(5, NX * NY * NZ * 1);

    // init diffeo = identity
    auto diffeoKernel = DiffeoKernel().create_kernel("diffeo");
    diffeoKernel.set_arg(0, diffeo.field);
    diffeoKernel.set_arg(1, NX);
    diffeoKernel.set_arg(2, NX * NY);
    diffeoKernel.set_arg(3, NX * NY * NZ * 1);
    Queue_.enqueue_nd_range_kernel(diffeoKernel, 3, NULL, workDim, NULL);

    auto updateDiffeoKernel = UpdateDiffeoKernel().create_kernel("updateDiffeo");
    updateDiffeoKernel.set_arg(0, diffeo.field);
    updateDiffeoKernel.set_arg(1, velocity.field);
    updateDiffeoKernel.set_arg(2, NX);
    updateDiffeoKernel.set_arg(3, NX * NY);
    updateDiffeoKernel.set_arg(4, NX * NY * NZ * 1);
    updateDiffeoKernel.set_arg(5, deltaT);

    for (auto tau = 0; tau < N; ++tau) {
        // calculate nabla(I)
        CptGradScalarField(image, imageGrad, Queue_, 0, DeltaX_);

        // compute velocity field: v = -K (*) nabla(I)P
        for (auto dir = 0; dir < 3; ++dir) {
            velocityKernel.set_arg(6, dir);
            Queue_.enqueue_nd_range_kernel(velocityKernel, 3, NULL, workDim, NULL);
        }
        Convolver_.Convolution(velocity.field, 1);

        // estimate phi^{tau + 1} from phi^{tau} and v^{tau}
        Queue_.enqueue_nd_range_kernel(updateDiffeoKernel, 3, NULL, workDim, NULL);

        // project I^0 to time (tau + 1)
        ApplyMapping(Image_, diffeo, image, Queue_);

        // compute momentum
        auto cpuMomentum = CopyOnHost(momentum, Queue_);
        auto cpuVelocity = CopyOnHost(velocity, Queue_);
        std::vector<float> B(
            std::max({ NX, NY, NZ }),
            1.f / (3.f * deltaT)
        );

        // x coordinate
        std::vector<float> A(NX), C(NX), D(NX), X(NX);
        for (auto k = 0; k < NZ; ++k) {
            for (auto j = 0; j < NY; ++j) {
                A[0] = 0;
                for (auto i = 0; i < NX - 1; ++i) {
                    A[i + 1] = -cpuVelocity.G<0>(i, j, k) / (2.f * DeltaX_);
                    C[i] = cpuVelocity.G<0>(i + 1, j, k) / (2.f * DeltaX_);
                    D[i] = cpuMomentum.G<0>(i, j, k) / (3.f * deltaT);
                }
                C[NX - 1] = 0;
                D[NX - 1] = cpuMomentum.G<0>(NX - 1, j, k);
                TridiagonalSolveFloat(&A[0], &B[0], &C[0], &D[0], &X[0], NX);
                for (auto i = 0; i < NX; ++i)
                    cpuMomentum.P({ X[i] }, i, j, k);
            }
        }

        // y coordinate
        A.resize(NY);
        C.resize(NY);
        D.resize(NY);
        X.resize(NY);
        for (auto k = 0; k < NZ; ++k) {
            for (auto i = 0; i < NX; ++i) {
                A[0] = 0;
                for (auto j = 0; j < NY - 1; ++j) {
                    A[j + 1] = -cpuVelocity.G<1>(i, j, k) / (2.f * DeltaX_);
                    C[j] = cpuVelocity.G<1>(i, j + 1, k) / (2.f * DeltaX_);
                    D[j] = cpuMomentum.G<0>(i, j, k) / (3.f * deltaT);
                }
                C[NY - 1] = 0;
                D[NY - 1] = cpuMomentum.G<0>(i, NY - 1, k);
                TridiagonalSolveFloat(&A[0], &B[0], &C[0], &D[0], &X[0], NY);
                for (auto j = 0; j < NY; ++j)
                    cpuMomentum.P({ X[j] }, i, j, k);
            }
        }

        // z coordinate
        A.resize(NZ);
        C.resize(NZ);
        D.resize(NZ);
        X.resize(NZ);
        for (auto j = 0; j < NY; ++j) {
            for (auto i = 0; i < NX; ++i) {
                A[0] = 0;
                for (auto k = 0; k < NZ - 1; ++k) {
                    A[k + 1] = -cpuVelocity.G<2>(i, j, k) / (2.f * DeltaX_);
                    C[k] = cpuVelocity.G<2>(i, j, k + 1) / (2.f * DeltaX_);
                    D[k] = cpuMomentum.G<0>(i, j, k) / (3.f * deltaT);
                }
                C[NZ - 1] = 0;
                D[NZ - 1] = cpuMomentum.G<0>(i, j, NZ - 1);
                TridiagonalSolveFloat(&A[0], &B[0], &C[0], &D[0], &X[0], NZ);
                for (auto k = 0; k < NZ; ++k)
                    cpuMomentum.P({ X[k] }, i, j, k);
            }
        }

        // copy back to GPU
        compute::copy(cpuMomentum.Begin(), cpuMomentum.End(), momentum.field.begin(), Queue_);
        break;
    }

    auto v = CopyOnHost(velocity, Queue_);
    v.Write({"/Users/alexm/Desktop/v_x.nii", "/Users/alexm/Desktop/v_y.nii", "/Users/alexm/Desktop/v_z.nii"});

    auto i = CopyOnHost(image, Queue_);
    i.Write({"/Users/alexm/Desktop/final_im.nii"});

    auto mom = CopyOnHost(momentum, Queue_);
    mom.Write({"/Users/alexm/Desktop/final_mom.nii"});
}