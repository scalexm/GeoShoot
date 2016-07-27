/*=========================================================================
 
 FFTConvolver.hpp
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

#ifndef FFT_CONVOLVER_HPP
#define FFT_CONVOLVER_HPP

#include "VectorField.hpp"
#include <clFFT/clFFT.h>

// non-copyable, non-movable
class FFTConvolver {
private:
    clfftPlanHandle PlanHandle_ = 0;

    compute::vector<float> Filter_, Signal_;
    compute::command_queue Queue_;

    int NXfft_, NYfft_, NZfft_;

    void MakeSumOf7AnisotropicGaussianFilters(
        const std::array<float, 7> & weights = { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        const std::array<float, 7> & sigmaXs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaYs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaZs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        bool normalizeWeights = true
    );

    compute::int4_ Dims() const {
        return { NXfft_, NYfft_, NZfft_, 0 };
    }

public:
    FFTConvolver(compute::command_queue);
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

    // in place convolution
    void Convolution(GPUVectorField<3> & field);
};

#endif
