//
//  FFTConvolver.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef FFT_CONVOLVER_HPP
#define FFT_CONVOLVER_HPP

#include <clFFT/clFFT.h>
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>

namespace compute = boost::compute;

// non-copyable, non-movable
class FFTConvolver {
private:
    clfftPlanHandle PlanHandle_ = 0;
    compute::vector<float> Filter_, Signal_;
    compute::command_queue Queue_;

    int NX_, NY_, NZ_, NXfft_, NYfft_, NZfft_;
    int ComplexPadding_;

    void MakeSumOf7AnisotropicGaussianFilters(
        const std::array<float, 7> & weights = { 100.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        const std::array<float, 7> & sigmaXs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaYs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaZs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        bool normalizeWeights = true
    );

public:
    FFTConvolver();
    FFTConvolver(const FFTConvolver &) = delete;
    FFTConvolver(FFTConvolver &&) = delete;
    FFTConvolver & operator =(const FFTConvolver &) = delete;
    FFTConvolver & operator =(FFTConvolver &&) = delete;
    ~FFTConvolver();

    void InitiateConvolver(
        int NX, int NY, int NZ,
        const std::array<float, 7> & weights = { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        const std::array<float, 7> & sigmaXs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaYs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaZs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f }
    );

    void ChangeKernel(
        const std::array<float, 7> & weights = { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        const std::array<float, 7> & sigmaXs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaYs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaZs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f }
    );

    void Convolution(compute::vector<float> & field, int NT);
};

#endif
