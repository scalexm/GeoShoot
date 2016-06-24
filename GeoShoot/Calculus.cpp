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
                                   int NX, int NXtY, int NXtYtZ, int NXtYtZtT_grad, float deltaX,
                                   int tField, int tGrad) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NXtY;
                int ind_tField = ind + tField * NXtYtZ;
                int ind_tGrad = ind + tGrad * NXtYtZ;

                grad[ind_tGrad]
                    = (field[ind_tField + 1] - field[ind_tField - 1]) / (2.f * deltaX);
                grad[ind_tGrad + NXtYtZtT_grad]
                    = (field[ind_tField + NX] - field[ind_tField - NX]) / (2.f * deltaX);
                grad[ind_tGrad + 2 * NXtYtZtT_grad]
                    = (field[ind_tField + NXtY] - field[ind_tField - NXtY]) / (2.f * deltaX);
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void CptGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                        compute::command_queue & queue,
                        int timeFrame, float deltaX) {
    assert(field.NX == gradient.NX && field.NY == gradient.NY && field.NZ == gradient.NZ);
    assert(timeFrame >= 0 || field.NT == gradient.NT);

    auto startTime = timeFrame == -1 ? 0 : timeFrame;
    auto endTime = timeFrame == -1 ? field.NT : (timeFrame + 1);

    auto kernel = GradientKernel().create_kernel("gradient");
    size_t workDim[3] = {
        (size_t) field.NX - 1,
        (size_t) field.NY - 1,
        (size_t) (field.NZ == 1 ? 1 : (field.NZ - 1)) // handle 2D images correctly
    };
    size_t workOffset[3] = { 1, 1, (size_t) (field.NZ == 1 ? 0 : 1) };
    kernel.set_arg(0, gradient.field);
    kernel.set_arg(1, field.field);
    kernel.set_arg(2, field.NX);
    kernel.set_arg(3, field.NX * field.NY);
    kernel.set_arg(4, field.NX * field.NY * field.NZ);
    kernel.set_arg(5, field.NX * field.NY * field.NZ * gradient.NT);
    kernel.set_arg(6, deltaX);

    // fill gradient with 0.f, so that boundaries will be at 0.f
    compute::fill(gradient.field.begin(), gradient.field.end(), 0.f, queue);

    for (auto t = startTime; t < endTime; ++t) {
        int tGrad = timeFrame == -1 ? t : 0;

        kernel.set_arg(7, t);
        kernel.set_arg(8, tGrad);
        queue.enqueue_nd_range_kernel(kernel, 3, workOffset, workDim, NULL);
    }
}

namespace {
    static std::string InterpSource = R"#(
        float interp(__global const float * field, int dir,
                     float x, float y, float z, int t,
                     int NX, int NY, int NZ, int NT) {
            int NXtY = NX * NY;
            int NXtYtZ = NXtY * NZ;
            int NXtYtZtT = NXtYtZ * NT;

            if (x < 0.) x = 0.0001;
            if (x >= NX - 1.) x = NX - 1.0001;
            if (y < 0.) y = 0.0001;
            if (y >= NY - 1.) y = NY - 1.0001;
            if (z < 0.) z = 0.0001;
            if (z >= NZ - 1.) z = NZ - 1.0001;
            if (t < 0) t = 0;
            if (t > NT - 1) t = NT - 1;

            int xi = (int)x; float xwm = 1 - (x - (float)xi); float xwp = x - (float)xi;
            int yi = (int)y; float ywm = 1 - (y - (float)yi); float ywp = y - (float)yi;
            int zi = (int)z; float zwm = 1 - (z - (float)zi); float zwp = z - (float)zi;

            int ind = dir * NXtYtZtT + t * NXtYtZ;
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

    compute::program & MappingKernel() {
        static std::string source = InterpSource + R"#(
            __kernel void mapping(__global float * dst, __global const float * src,
                                  __global const float * map, int t,
                                  int NX, int NY, int NZ, int NT) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int NXtY = NX * NY;
                int NXtYtZ = NXtY * NZ;
                int NXtYtZtT = NXtYtZ * NT;
                int ind = x + y * NX + z * NXtY;
                int ind_t = ind + t * NXtYtZ;

                dst[ind] = interp(
                    src,
                    0,
                    map[ind_t],
                    map[ind_t + NXtYtZtT],
                    map[ind_t + 2 * NXtYtZtT],
                    0,
                    NX, NY, NZ, 1
                );
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void ApplyMapping(const GPUScalarField & src, const GPUVectorField<3> & mapping,
                  GPUScalarField & dst, compute::command_queue & queue, int t) {
    assert(src.NX == mapping.NX && src.NY == mapping.NY && src.NZ == mapping.NZ);
    assert(src.NX == dst.NX && src.NY == dst.NY && src.NZ == dst.NZ);
    assert(src.NT == 1 && dst.NT == 1);

    auto kernel = MappingKernel().create_kernel("mapping");
    size_t workDim[3] = { (size_t) src.NX, (size_t) src.NY, (size_t) src.NZ };
    kernel.set_arg(0, dst.field);
    kernel.set_arg(1, src.field);
    kernel.set_arg(2, mapping.field);
    kernel.set_arg(3, t);
    kernel.set_arg(4, src.NX);
    kernel.set_arg(5, src.NY);
    kernel.set_arg(6, src.NZ);
    kernel.set_arg(7, src.NT);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}
