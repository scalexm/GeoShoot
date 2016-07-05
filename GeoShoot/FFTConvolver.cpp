//
//  FFTConvolver.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include "FFTConvolver.hpp"
#include "GPU.hpp"

#define CHECK_ERROR(err) if (err != 0) { throw compute::opencl_error { err }; }

FFTConvolver::FFTConvolver(compute::command_queue queue) : Queue_ { std::move(queue) } {
}

FFTConvolver::~FFTConvolver() {
    if (PlanHandle_)
        clfftDestroyPlan(&PlanHandle_);
}

void FFTConvolver::InitiateConvolver(
    int NX, int NY, int NZ,
    const std::array<float, 7> & weights,
    const std::array<float, 7> & sigmaXs,
    const std::array<float, 7> & sigmaYs,
    const std::array<float, 7> & sigmaZs
) {

    //smaller size higher than 'this->NX' and being a power of 2
    NXfft_ = (int)(pow(2., floor(log(NX) / log(2.) + 0.99999)) + 0.00001);
    NYfft_ = (int)(pow(2., floor(log(NY) / log(2.) + 0.99999)) + 0.00001);
    NZfft_ = (int)(pow(2., floor(log(NZ) / log(2.) + 0.99999)) + 0.00001);

    Filter_ = compute::vector<float>(2 * NXfft_  * NYfft_ * NZfft_, Queue_.get_context());
    Signal_ = compute::vector<float>(2 * NXfft_  * NYfft_ * NZfft_, Queue_.get_context());

    if (PlanHandle_)
        clfftDestroyPlan(&PlanHandle_);

    clfftDim dim = CLFFT_3D;
    size_t clLengths[3] = { (size_t) NXfft_, (size_t) NYfft_, (size_t) NZfft_ };

    auto err = clfftCreateDefaultPlan(&PlanHandle_, Queue_.get_context(), dim, clLengths);
    CHECK_ERROR(err);
    err = clfftSetPlanPrecision(PlanHandle_, CLFFT_SINGLE);
    CHECK_ERROR(err);
    err = clfftSetLayout(PlanHandle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    CHECK_ERROR(err);
    err = clfftSetResultLocation(PlanHandle_, CLFFT_INPLACE);
    CHECK_ERROR(err);
    err = clfftBakePlan(PlanHandle_, 1, &Queue_.get(), NULL, NULL);
    CHECK_ERROR(err);

    ChangeKernel(weights, sigmaXs, sigmaYs, sigmaZs);
        
}

void FFTConvolver::ChangeKernel(
    const std::array<float, 7> & weights,
    const std::array<float, 7> & sigmaXs,
    const std::array<float, 7> & sigmaYs,
    const std::array<float, 7> & sigmaZs
) {
    MakeSumOf7AnisotropicGaussianFilters(weights, sigmaXs, sigmaYs, sigmaZs, true);
}

namespace {
    compute::program & GaussianKernel() {
        static std::string source = R"#(
            __kernel void gaussian(__global float * data, int NX, int NY, int NZ,
                                   float sigmaX, float sigmaY, float sigmaZ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int ind = x + y * NX + z * NX * NZ;

                if (x >= NX / 2)
                    x -= NX;
                if (y >= NY / 2)
                    y -= NY;
                if (z >= NZ / 2)
                    z -= NZ;
                data[ind] = exp(
                    -x * x / (2. * sigmaX * sigmaX)
                    -y * y / (2. * sigmaY * sigmaY)
                    -z * z / (2. * sigmaZ * sigmaZ)
                );
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program & AddKernel() {
        static std::string source = R"#(
            __kernel void add(__global float * filter, __global float * temp, float coeff) {
                int ind = get_global_id(0);
                filter[2 * ind] += temp[ind] * coeff;
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void FFTConvolver::MakeSumOf7AnisotropicGaussianFilters(
    const std::array<float, 7> & weights,
    const std::array<float, 7> & sigmaXs,
    const std::array<float, 7> & sigmaYs,
    const std::array<float, 7> & sigmaZs,
    bool normalizeWeights
) {
    assert(PlanHandle_ != 0);
    compute::vector<float> temp(Filter_.size() / 2, Queue_.get_context());
    compute::fill(Filter_.begin(), Filter_.end(), 0.f, Queue_);

    auto gaussianKernel = GaussianKernel().create_kernel("gaussian");
    size_t workDim[3] = { (size_t) NXfft_, (size_t) NYfft_, (size_t) NZfft_ };
    gaussianKernel.set_arg(0, temp);
    gaussianKernel.set_arg(1, NXfft_);
    gaussianKernel.set_arg(2, NYfft_);
    gaussianKernel.set_arg(3, NZfft_);

    auto addKernel = AddKernel().create_kernel("add");
    size_t addWorkDim[1] = { (size_t) NXfft_ * NYfft_ * NZfft_ };
    addKernel.set_arg(0, Filter_);
    addKernel.set_arg(1, temp);

    using boost::compute::_1;

    auto sumWeight = 0.f;
    for (auto k = 0; k < 7; ++k) {
        sumWeight += weights[k];
        compute::fill(temp.begin(), temp.end(), 0.f, Queue_);
        gaussianKernel.set_arg(4, sigmaXs[k]);
        gaussianKernel.set_arg(5, sigmaYs[k]);
        gaussianKernel.set_arg(6, sigmaZs[k]);
        Queue_.enqueue_nd_range_kernel(gaussianKernel, 3, NULL, workDim, NULL);

        auto sumLoc = 0.f;
        compute::reduce(temp.begin(), temp.end(), &sumLoc, compute::plus<float>(), Queue_);
        if (fabs(sumLoc) > 0.01) {
            auto coeff = weights[k] / sumLoc;
            addKernel.set_arg(2, coeff);
            Queue_.enqueue_nd_range_kernel(addKernel, 1, NULL, addWorkDim, NULL);
        } else
            std::cout << "One kernel appears to be null or almost null" << std::endl;
    }

    if (normalizeWeights)
        compute::transform(Filter_.begin(), Filter_.end(), Filter_.begin(), _1 / sumWeight, Queue_);

    auto err = clfftEnqueueTransform(
        PlanHandle_,
        CLFFT_FORWARD,
        1,
        &Queue_.get(),
        0,
        NULL,
        NULL,
        &Filter_.get_buffer().get(),
        NULL,
        NULL
    );

    if (err != 0)
        throw compute::opencl_error(err);
}

namespace {
    compute::program & CopyKernel() {
        static std::string source = R"#(
            __kernel void copy(__global float * out, __global const float * in,
                               int NX_out, int NXtY_out, int NXtYtZ_out,
                               int NX_in, int NXtY_in, int NXtYtZ_in,
                               int dirOut, int dirIn,
                               int strideOut, int strideIn) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                int indOut = x + y * NX_out + z * NXtY_out + dirOut * NXtYtZ_out;
                int indIn = x + y * NX_in + z * NXtY_in + dirIn * NXtYtZ_in;
                out[strideOut * indOut] = in[strideIn * indIn];
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }

    compute::program & FilterKernel() {
        static std::string source = R"#(
            __kernel void filter(__global float * signal, __global const float * filter) {
                int ind = 2 * get_global_id(0);
                float a = signal[ind];
                float b = signal[ind + 1];
                float c = filter[ind];
                float d = filter[ind + 1];

                signal[ind] = a * c - b * d;
                signal[ind + 1] = c * b + a * d;
            }
        )#";

        MAKE_PROGRAM(source, GetContext());
    }
}

void FFTConvolver::Convolution(GPUVectorField<3> & field) {
    assert(field.NX() <= NXfft_);
    assert(field.NY() <= NYfft_);
    assert(field.NZ() <= NZfft_);

    // copy between field and Signal_ (a signal which will be processed by clFFT), complying
    // with clFFT complex interleaved layout
    auto copyKernel = CopyKernel().create_kernel("copy");
    size_t workDim[3] = { (size_t) field.NX(), (size_t) field.NY(), (size_t) field.NZ() };

    auto filterKernel = FilterKernel().create_kernel("filter");
    size_t filterWorkDim[1] = { (size_t) NXfft_ * NYfft_ * NZfft_ };
    filterKernel.set_arg(0, Signal_);
    filterKernel.set_arg(1, Filter_);

    // field maps to a 3D space
    for (auto dir = 0; dir < 3; ++dir) {
        compute::fill(Signal_.begin(), Signal_.end(), 0.f, Queue_);
        // copy <field | e_dir> to signal
        copyKernel.set_arg(0, Signal_);
        copyKernel.set_arg(1, field.Buffer());
        copyKernel.set_arg(2, NXfft_);
        copyKernel.set_arg(3, NXfft_ * NYfft_);
        copyKernel.set_arg(4, 0);
        copyKernel.set_arg(5, field.NX());
        copyKernel.set_arg(6, field.NY() * field.NZ());
        copyKernel.set_arg(7, field.NZ() * field.NY() * field.NZ());
        copyKernel.set_arg(8, 0);
        copyKernel.set_arg(9, dir);
        copyKernel.set_arg(10, 2);
        copyKernel.set_arg(11, 1);
        Queue_.enqueue_nd_range_kernel(copyKernel, 3, NULL, workDim, NULL);

        // process FFT
        auto err = clfftEnqueueTransform(
            PlanHandle_,
            CLFFT_FORWARD,
            1,
            &Queue_.get(),
            0,
            NULL,
            NULL,
            &Signal_.get_buffer().get(),
            NULL,
            NULL
        );
        CHECK_ERROR(err);

        // apply filter
        Queue_.enqueue_nd_range_kernel(filterKernel, 1, NULL, filterWorkDim, NULL);

        // process IFFT
        err = clfftEnqueueTransform(
            PlanHandle_,
            CLFFT_BACKWARD,
            1,
            &Queue_.get(),
            0,
            NULL,
            NULL,
            &Signal_.get_buffer().get(),
            NULL,
            NULL
        );
        CHECK_ERROR(err);

        // copy back signal to <field | e_dir>
        copyKernel.set_arg(0, field.Buffer());
        copyKernel.set_arg(1, Signal_);
        copyKernel.set_arg(2, field.NX());
        copyKernel.set_arg(3, field.NX() * field.NY());
        copyKernel.set_arg(4, field.NX() * field.NY() * field.NZ());
        copyKernel.set_arg(5, NXfft_);
        copyKernel.set_arg(6, NXfft_ * NYfft_);
        copyKernel.set_arg(7, 0);
        copyKernel.set_arg(8, dir);
        copyKernel.set_arg(9, 0);
        copyKernel.set_arg(10, 1);
        copyKernel.set_arg(11, 2);
        Queue_.enqueue_nd_range_kernel(copyKernel, 3, NULL, workDim, NULL);
    }
}