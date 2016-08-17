/*=========================================================================
 
 FFTConvolver.cpp
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

#include "FFTConvolver.hpp"

FFTConvolver::FFTConvolver(compute::command_queue queue) : Queue_ { std::move(queue) } {
}

FFTConvolver::FFTConvolver(FFTConvolver && that) {
    *this = std::move(that);
}

FFTConvolver & FFTConvolver::operator =(FFTConvolver && that) {
    PlanHandle_ = that.PlanHandle_;
    Queue_ = std::move(that.Queue_);
    Filter_ = std::move(that.Filter_);
    NXfft_ = that.NXfft_;
    NYfft_ = that.NYfft_;
    NZfft_ = that.NZfft_;
    return *this;
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
    const std::array<float, 7> & sigmaZs,
    bool normalizeWeights
) {

    //smaller size higher than 'this->NX' and being a power of 2
    NXfft_ = (int)(pow(2., floor(log(NX) / log(2.) + 0.99999)) + 0.00001);
    NYfft_ = (int)(pow(2., floor(log(NY) / log(2.) + 0.99999)) + 0.00001);
    NZfft_ = (int)(pow(2., floor(log(NZ) / log(2.) + 0.99999)) + 0.00001);

    Filter_ = compute::vector<float>(2 * NXfft_ * NYfft_ * NZfft_, Queue_.get_context());

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

    ChangeKernel(weights, sigmaXs, sigmaYs, sigmaZs, normalizeWeights);
        
}

void FFTConvolver::ChangeKernel(
    const std::array<float, 7> & weights,
    const std::array<float, 7> & sigmaXs,
    const std::array<float, 7> & sigmaYs,
    const std::array<float, 7> & sigmaZs,
    bool normalizeWeights
) {
    MakeSumOf7AnisotropicGaussianFilters(weights, sigmaXs, sigmaYs, sigmaZs, normalizeWeights);
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

    auto gaussianKernel = GetProgram().create_kernel("gaussian");
    size_t workDim[3] = { (size_t) NXfft_, (size_t) NYfft_, (size_t) NZfft_ };
    gaussianKernel.set_arg(0, temp);
    gaussianKernel.set_arg(1, Dims());

    auto addKernel = GetProgram().create_kernel("addFFT");
    size_t addWorkDim[1] = { (size_t) NXfft_ * NYfft_ * NZfft_ };
    addKernel.set_arg(0, Filter_);
    addKernel.set_arg(1, temp);

    auto sumWeight = 0.f;
    for (auto k = 0; k < 7; ++k) {
        sumWeight += weights[k];
        compute::fill(temp.begin(), temp.end(), 0.f, Queue_);
        gaussianKernel.set_arg(2, sigmaXs[k]);
        gaussianKernel.set_arg(3, sigmaYs[k]);
        gaussianKernel.set_arg(4, sigmaZs[k]);
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

    using boost::compute::_1;

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

    CHECK_ERROR(err);
}