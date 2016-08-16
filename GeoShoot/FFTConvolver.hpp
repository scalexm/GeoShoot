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
#include "GPU.hpp"
#include <clFFT/clFFT.h>

#define CHECK_ERROR(err) if (err != 0) { throw compute::opencl_error { err }; }

// non-copyable
class FFTConvolver {
private:
    clfftPlanHandle PlanHandle_ = 0;

    compute::vector<float> Filter_;
    compute::command_queue Queue_;

    int NXfft_, NYfft_, NZfft_;

    void MakeSumOf7AnisotropicGaussianFilters(
        const std::array<float, 7> & weights = { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        const std::array<float, 7> & sigmaXs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaYs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        const std::array<float, 7> & sigmaZs = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        bool normalizeWeights = true
    );

public:
    FFTConvolver(compute::command_queue);
    FFTConvolver(const FFTConvolver &) = delete;
    FFTConvolver & operator =(const FFTConvolver &) = delete;

    FFTConvolver(FFTConvolver &&);
    FFTConvolver & operator =(FFTConvolver &&);
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

    compute::int4_ Dims() const {
        return { NXfft_, NYfft_, NZfft_, 0 };
    }

    // in place convolution
    template<size_t Dim>
    void Convolution(GPUVectorField<Dim> & field, GPUVectorField<2> & acc) {
        assert(field.NX() <= NXfft_);
        assert(field.NY() <= NYfft_);
        assert(field.NZ() <= NZfft_);

        // copy between field and Signal_ (a signal which will be processed by clFFT), complying
        // with clFFT complex interleaved layout
        auto copyKernel = GetProgram().create_kernel("copyFFT");
        size_t workDim[3] = { (size_t) field.NX(), (size_t) field.NY(), (size_t) field.NZ() };

        auto filterKernel = GetProgram().create_kernel("filter");
        size_t filterWorkDim[1] = { (size_t) NXfft_ * NYfft_ * NZfft_ };
        filterKernel.set_arg(0, acc.Buffer());
        filterKernel.set_arg(1, Filter_);

        // field maps to a 3D space
        for (auto dir = 0; dir < Dim; ++dir) {
            acc.Fill(0.f, Queue_);

            // copy <field | e_dir> to signal
            copyKernel.set_arg(0, acc.Buffer());
            copyKernel.set_arg(1, field.Buffer());
            copyKernel.set_arg(2, NXfft_);
            copyKernel.set_arg(3, NXfft_ * NYfft_);
            copyKernel.set_arg(4, 0);
            copyKernel.set_arg(5, field.NX());
            copyKernel.set_arg(6, field.NX() * field.NY());
            copyKernel.set_arg(7, field.NX() * field.NY() * field.NZ());
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
                &acc.Buffer().get_buffer().get(),
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
                &acc.Buffer().get_buffer().get(),
                NULL,
                NULL
            );
            CHECK_ERROR(err);

            // copy back signal to <field | e_dir>
            copyKernel.set_arg(0, field.Buffer());
            copyKernel.set_arg(1, acc.Buffer());
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
};

#endif
