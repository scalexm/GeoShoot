//
//  Calculus.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 24/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include "Calculus.hpp"
#include "GPU.hpp"

namespace {
    compute::program & GradientKernel() {
        static std::string source = R"#(
            __kernel void gradient(__global float * grad, __global const float * field,
                                   int NX, int NY, int NZ, float deltaX) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NX * NY * NZ;
                int ind = x + y * NX + z * NXtY;

                if (x == 0 || y == 0 || z == 0 || x == NX - 1 || y == NY - 1 || z == NZ - 1) {
                    grad[ind] = 0.f;
                    grad[ind + NXtYtZ] = 0.f;
                    grad[ind + 2 * NXtYtZ] = 0.f;
                } else {
                    grad[ind]
                        = (field[ind + 1] - field[ind - 1]) / (2.f * deltaX);
                    grad[ind + NXtYtZ]
                        = (field[ind + NX] - field[ind - NX]) / (2.f * deltaX);
                    grad[ind + 2 * NXtYtZ]
                        = (field[ind + NXtY] - field[ind - NXtY]) / (2.f * deltaX);
                }
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void CptGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                        float deltaX, compute::command_queue & queue) {
    assert(field.NX() == gradient.NX());
    assert(field.NY() == gradient.NY());
    assert(field.NZ() == gradient.NZ());

    auto kernel = GradientKernel().create_kernel("gradient");
    size_t workDim[3] = { (size_t) field.NX(), (size_t) field.NY(), (size_t) field.NZ() };
    kernel.set_arg(0, gradient.Buffer());
    kernel.set_arg(1, field.Buffer());
    kernel.set_arg(2, field.NX());
    kernel.set_arg(3, field.NY());
    kernel.set_arg(4, field.NZ());
    kernel.set_arg(5, deltaX);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

namespace {
    static std::string InterpSource = R"#(
        float interp(__global const float * field, int dir,
                     float x, float y, float z,
                     int NX, int NY, int NZ) {
            int NXtY = NX * NY;
            int NXtYtZ = NXtY * NZ;

            if (x < 0.) x = 0.0001;
            if (x >= NX - 1.) x = NX - 1.0001;
            if (y < 0.) y = 0.0001;
            if (y >= NY - 1.) y = NY - 1.0001;
            if (z < 0.) z = 0.0001;
            if (z >= NZ - 1.) z = NZ - 1.0001;

            int xi = (int)x; float xwm = 1 - (x - (float)xi); float xwp = x - (float)xi;
            int yi = (int)y; float ywm = 1 - (y - (float)yi); float ywp = y - (float)yi;
            int zi = (int)z; float zwm = 1 - (z - (float)zi); float zwp = z - (float)zi;

            int ind = dir * NXtYtZ;
            float interpoGreyLevel;
            if (NZ == 1) { //2D IMAGE
                float wmm = xwm * ywm;
                float wmp = xwm * ywp;
                float wpm = xwp * ywm;
                float wpp = xwp * ywp;

                interpoGreyLevel = wmm * field[ind + yi * NX + (xi)];
                interpoGreyLevel += wmp * field[ind + (yi + 1) * NX + (xi)];
                interpoGreyLevel += wpm * field[ind + yi * NX + xi + 1];
                interpoGreyLevel += wpp * field[ind + (yi + 1) * NX + xi + 1];
            } else { //3D IMAGE
                float wmmm = xwm * ywm * zwm, wmmp = xwm * ywm * zwp, wmpm = xwm * ywp * zwm,
                      wmpp = xwm * ywp * zwp;
                float wpmm = xwp * ywm * zwm, wpmp = xwp * ywm * zwp, wppm = xwp * ywp * zwm,
                      wppp = xwp * ywp * zwp;

                interpoGreyLevel = wmmm * field[ind + zi * NXtY + yi * NX + xi];
                interpoGreyLevel += wpmm * field[ind + zi * NXtY + yi * NX + xi + 1];
                interpoGreyLevel += wmpm * field[ind + zi * NXtY + (yi + 1) * NX + xi];
                interpoGreyLevel += wppm * field[ind + zi * NXtY + (yi + 1) * NX + xi + 1];
                interpoGreyLevel += wmmp * field[ind + (zi + 1) * NXtY + yi * NX + xi];
                interpoGreyLevel += wpmp * field[ind + (zi + 1) * NXtY + yi * NX + xi + 1];
                interpoGreyLevel += wmpp * field[ind + (zi + 1) * NXtY + (yi + 1) * NX + xi];
                interpoGreyLevel += wppp * field[ind + (zi + 1) * NXtY + (yi + 1) * NX + xi + 1];
            }

            return interpoGreyLevel;
        }
    )#";

    compute::program & TransportImageKernel() {
        static std::string source = InterpSource + R"#(
            __kernel void transportImage(__global float * dst, __global const float * src,
                                         __global const float * diffeo,
                                         int NX, int NY, int NZ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int ind = x + y * NX + z * NXtY;

                dst[ind] = interp(
                    src,
                    0,
                    diffeo[ind],
                    diffeo[ind + NXtYtZ],
                    diffeo[ind + 2 * NXtYtZ],
                    NX, NY, NZ
                );
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program TransportMomentumKernel() {
        static std::string source = InterpSource + R"#(
            __kernel void transportMomentum(__global float * dst, __global const float * src,
                                            __global const float * diffeo,
                                            int NX, int NY, int NZ, float deltaX) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int ind = x + y * NX + z * NXtY;

                if (x == 0 || y == 0 || z == 0 || x == NX - 1 || y == NY - 1 || z == NZ - 1)
                    dst[ind] = src[ind];
                else {
                    float twoDelta = 2.f * deltaX;
                    float d11 = (diffeo[ind + 1] - diffeo[ind - 1]) / twoDelta;
                    float d12 = (diffeo[ind + NX] - diffeo[ind - NX]) / twoDelta;
                    float d13 = (diffeo[ind + NXtY] - diffeo[ind - NXtY]) / twoDelta;
                    float xx = diffeo[ind];

                    int iind = ind + NXtYtZ;
                    float d21 = (diffeo[iind + 1] - diffeo[iind - 1]) / twoDelta;
                    float d22 = (diffeo[iind + NX] - diffeo[iind - NX]) / twoDelta;
                    float d23 = (diffeo[iind + NXtY] - diffeo[iind - NXtY]) / twoDelta;
                    float yy = diffeo[iind];

                    iind += NXtYtZ;
                    float d31 = (diffeo[iind + 1] - diffeo[iind - 1]) / twoDelta;
                    float d32 = (diffeo[iind + NX] - diffeo[iind - NX]) / twoDelta;
                    float d33 = (diffeo[iind + NXtY] - diffeo[iind - NXtY]) / twoDelta;
                    float zz = diffeo[iind];

                    float val = interp(src, 0, xx, yy, zz, NX, NY, NZ);

                    // Jacobian
                    dst[ind] =
                        val*(d11*(d22*d33-d32*d23)-d21*(d12*d33-d32*d13)+d31*(d12*d23-d22*d13));
                }
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void TransportImage(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                    GPUScalarField & dst, compute::command_queue & queue) {
    assert(src.NX() == diffeo.NX());
    assert(src.NY() == diffeo.NY());
    assert(src.NZ() == diffeo.NZ());
    assert(dst.NX() == src.NX());
    assert(dst.NY() == src.NY());
    assert(dst.NZ() == src.NZ());

    auto kernel = TransportImageKernel().create_kernel("transportImage");
    size_t workDim[3] = { (size_t) src.NX(), (size_t) src.NY(), (size_t) src.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src.Buffer());
    kernel.set_arg(2, diffeo.Buffer());
    kernel.set_arg(3, src.NX());
    kernel.set_arg(4, src.NY());
    kernel.set_arg(5, src.NZ());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void TransportMomentum(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                       GPUScalarField & dst, float deltaX, compute::command_queue & queue) {
    assert(src.NX() == diffeo.NX());
    assert(src.NY() == diffeo.NY());
    assert(src.NZ() == diffeo.NZ());
    assert(dst.NX() == src.NX());
    assert(dst.NY() == src.NY());
    assert(dst.NZ() == src.NZ());

    auto kernel = TransportMomentumKernel().create_kernel("transportMomentum");
    size_t workDim[3] = { (size_t) src.NX(), (size_t) src.NY(), (size_t) src.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src.Buffer());
    kernel.set_arg(2, diffeo.Buffer());
    kernel.set_arg(3, src.NX());
    kernel.set_arg(4, src.NY());
    kernel.set_arg(5, src.NZ());
    kernel.set_arg(6, deltaX);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

namespace {
    compute::program & UpdateDiffeoKernel() {
        static std::string source = InterpSource + R"#(
            __kernel void updateDiffeo(__global float * diffeo, __global const float * velocity,
                                       int NX, int NY, int NZ, float deltaT) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int ind = x + y * NX + z * NXtY;

                float xx = diffeo[ind];
                float yy = diffeo[ind + NXtYtZ];
                float zz = diffeo[ind + 2 * NXtYtZ];

                diffeo[ind] += deltaT * interp(velocity, 0, xx, yy, zz, NX, NY, NZ);
                diffeo[ind + NXtYtZ] += deltaT * interp(velocity, 1, xx, yy, zz, NX, NY, NZ);
                diffeo[ind + 2 * NXtYtZ] += deltaT * interp(velocity, 2, xx, yy, zz, NX, NY, NZ);
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program & UpdateInvDiffeoKernel() {
        static std::string source = InterpSource + R"#(
            float limiter(float a, float b) {
                if (a * b > 0.f)
                    return a < b ? a : b;
                return 0.f;
            }

            float scheme(__global const float * src,
                         int ind, int offset, float eta) {
                float deltaB = src[ind] - src[ind - offset];
                float deltaF = src[ind + offset] - src[ind];
                if (eta >= 0.f) {
                    float deltaBB = src[ind - offset] - src[ind - 2 * offset];
                    return -eta * (deltaB + 0.5f * (1.f - eta)
                        * (limiter(deltaB, deltaF) - limiter(deltaBB, deltaB)));
                } else {
                    float deltaFF = src[ind + 2 * offset] - src[ind + offset];
                    return eta * (-deltaF + 0.5f * (1.f + eta)
                        * (limiter(deltaFF, deltaF) - limiter(deltaF, deltaB)));
                }
            }

            __kernel void updateInvDiffeo(__global float * dst, __global const float * velocity,
                                          __global const float * src,
                                          int NX, int NY, int NZ,
                                          float deltaT, float deltaX) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int ind = x + y * NX + z * NXtY;

                if (x > 1 && y > 1 && z > 1 && x < NX - 2 && y < NY - 2 && z < NZ - 2) {
                    float ratio = deltaT / deltaX;
                    for (int dir = 0; dir < 3; ++dir) {
                        float val = 0.f;
                        int iind = ind + dir * NXtYtZ;

                        float eta = deltaT / deltaX * velocity[ind];
                        val += scheme(src, iind, 1, eta);

                        eta = deltaT / deltaX * velocity[ind + NXtYtZ];
                        val += scheme(src, iind, NX, eta);

                        eta = deltaT / deltaX * velocity[ind + 2 * NXtYtZ];
                        val += scheme(src, iind, NXtY, eta);

                        dst[iind] += val;
                    }
                }
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void UpdateDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                  compute::command_queue & queue) {
    assert(velocity.NX() == diffeo.NX());
    assert(velocity.NY() == diffeo.NY());
    assert(velocity.NZ() == diffeo.NZ());

    auto kernel = UpdateDiffeoKernel().create_kernel("updateDiffeo");
    size_t workDim[3] = { (size_t) diffeo.NX(), (size_t) diffeo.NY(), (size_t) diffeo.NZ() };
    kernel.set_arg(0, diffeo.Buffer());
    kernel.set_arg(1, velocity.Buffer());
    kernel.set_arg(2, diffeo.NX());
    kernel.set_arg(3, diffeo.NY());
    kernel.set_arg(4, diffeo.NZ());
    kernel.set_arg(5, deltaT);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void UpdateInvDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                     float deltaX, compute::command_queue & queue) {
    assert(velocity.NX() == diffeo.NX());
    assert(velocity.NY() == diffeo.NY());
    assert(velocity.NZ() == diffeo.NZ());

    auto temp = GPUVectorField<3> { diffeo.NX(), diffeo.NY(), diffeo.NZ(), queue.get_context() };
    compute::copy(diffeo.Begin(), diffeo.End(), temp.Begin(), queue);

    auto kernel = UpdateInvDiffeoKernel().create_kernel("updateInvDiffeo");
    size_t workDim[3] = { (size_t) diffeo.NX(), (size_t) diffeo.NY(), (size_t) diffeo.NZ() };
    kernel.set_arg(0, diffeo.Buffer());
    kernel.set_arg(1, velocity.Buffer());
    kernel.set_arg(2, temp.Buffer());
    kernel.set_arg(3, diffeo.NX());
    kernel.set_arg(4, diffeo.NY());
    kernel.set_arg(5, diffeo.NZ());
    kernel.set_arg(6, deltaT);
    kernel.set_arg(7, deltaX);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}