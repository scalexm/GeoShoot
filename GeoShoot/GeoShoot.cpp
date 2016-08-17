/*=========================================================================
 
 GeoShoot.cpp
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

#include "GeoShoot.hpp"
#include "GPU.hpp"
#include "Calculus.hpp"
#include "VectorField.hpp"
#include "Matrix.hpp"
#include <iostream>

/* GPU MEMORY TOTAL USAGE: 21 (GeoShoot) + 4 (FFT) scalar fields */
/* CPU MEMORY TOTAL USAGE: 2 * 3 * N scalar fields  */

GeoShoot::GeoShoot(const ScalarField & source, const ScalarField & target,
                   const ScalarField & momentum, ScalarField regions,
                   const Matrix<4, 4> & transfo, int N, compute::command_queue queue)
                   : N_ { std::max(2, N) },
                     InitialRegions_ { std::move(regions) },
                     Queue_ { queue } {
    if (source.NX() != momentum.NX()
        || source.NY() != momentum.NY()
        || source.NZ() != momentum.NZ()
        || source.NX() != target.NX()
        || source.NY() != target.NY()
        || source.NZ() != target.NZ()) {

        throw std::invalid_argument { "images do not have the same dimensions" };
    }

    NX_ = source.NX();
    NY_ = source.NY();
    NZ_ = source.NZ();

    DeltaT_ = 1.f / ((float) (N_ - 1));

    // fixed GPU fields
    Allocate(Source_);
    Allocate(InitialMomentum_);
    Allocate(Target_);

    if (InitialRegions_.NT() == 1)
        Allocate(GradientMomentum_);
    else
        GradientMomentum_ = GPUScalarField { NX_, NY_, NZ_, 2, GetContext() };

    // GPU fields used for computations
    Allocate(Scalar1_);
    Allocate(Scalar2_);
    Allocate(Scalar3_);
    Allocate(Scalar4_);
    Allocate(Scalar5_);
    Allocate(Vector1_);
    Allocate(Vector2_);
    Allocate(Vector3_);
    Allocate(Vector4_);

    Regions_ = GPUScalarField { NX_, NY_, NZ_, InitialRegions_.NT(), GetContext() };

    // used for Convolution with multiple regions
    if (Regions_.NT() > 1) {
        Allocate(Vector5_);
        Allocate(Vector6_);

        for (auto region = 0; region < Regions_.NT(); ++region) {
            InitialRegions_.ChangeChannel(region);
            Regions_.ChangeChannel(region);
            compute::copy(
                InitialRegions_.Begin(),
                InitialRegions_.End(),
                Regions_.Begin(),
                Queue_
            );
        }
    }

    Convolvers_.reserve(Regions_.NT());
    for (auto i = 0; i < Regions_.NT(); ++i)
        Convolvers_.emplace_back(Queue_); // in-place construction

    compute::copy(source.Begin(), source.End(), Source_.Begin(), Queue_);
    compute::copy(momentum.Begin(), momentum.End(), InitialMomentum_.Begin(), Queue_);
    compute::copy(target.Begin(), target.End(), Scalar1_ .Begin(), Queue_);

    auto template2TargetCoord = Invert4t4Quaternion(transfo);
    template2TargetCoord = Mult4t4Quaternion(template2TargetCoord, source.Image2World());
    template2TargetCoord = Mult4t4Quaternion(target.World2Image(), template2TargetCoord);

    std::cout << "\nTemplate to target:\n";
    for (auto && r : template2TargetCoord) {
        for (auto && c : r) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }

    ProjectImage(Scalar1_, Target_, template2TargetCoord, Queue_);

    auto && mat = source.Image2World();
    Xmm_ = sqrt(mat[0][0]*mat[0][0]+mat[0][1]*mat[0][1]+mat[0][2]*mat[0][2]);
    Ymm_ = sqrt(mat[1][0]*mat[1][0]+mat[1][1]*mat[1][1]+mat[1][2]*mat[1][2]);
    Zmm_ = sqrt(mat[2][0]*mat[2][0]+mat[2][1]*mat[2][1]+mat[2][2]*mat[2][2]);

    std::cout << "Template image resolution: " << Xmm_ << " "  << Ymm_ << " "  << Zmm_ << std::endl;

    if (fabs(Xmm_ - Ymm_) > 0.0001f || fabs(Xmm_ - Zmm_) > 0.0001f) {
        std::cout << "The image-to-world matrix of the source (template) image should be isotropic"
                  << " and this is not the case here!" << std::endl;
    }

    DiffeoTimeLine_ = VectorField<3> { NX_, NY_, NZ_, N_ - 1 };
    InvDiffeoTimeLine_ = VectorField<3> { NX_, NY_, NZ_, N_ - 1 };
}

void GeoShoot::Convolution(GPUVectorField<3> & field, GPUVectorField<3> & acc1,
                           GPUVectorField<3> & acc2) {
    if (Regions_.NT() == 1) {
        Convolvers_[0].Convolution(field, FFTAccumulator_);
        return;
    }

    acc1.Fill(0.f, Queue_);

    for (auto region = 0; region < Regions_.NT(); ++region) {
        Regions_.ChangeChannel(region);
        ScalarFieldTimesVectorField(Regions_, field, acc2, 1.f, Queue_);
        Convolvers_[region].Convolution(acc2, FFTAccumulator_);
        ScalarFieldTimesVectorField(Regions_, acc2, acc2, 1.f, Queue_);
        AddFields(acc1, acc2, acc1, 1.f, Queue_);
    }

    compute::copy(acc1.Begin(), acc1.End(), field.Begin(), Queue_);
}

void GeoShoot::Shooting() {
    Cost_ = Energy_ = 0.f;

    // init phi(0) = identity
    auto diffeoKernel = GetProgram().create_kernel("initDiffeo");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    diffeoKernel.set_arg(0, Vector1_.Buffer());
    diffeoKernel.set_arg(1, Dims());
    Queue_.enqueue_nd_range_kernel(diffeoKernel, 3, NULL, workDim, NULL); // Vector1_ <- phi(0)

    compute::copy(Vector1_.Begin(), Vector1_.End(), Vector2_.Begin(), Queue_); // Vector2_ <- phi^{-1}(0)

    // end of iteration tau: Vector1_ == phi(tau + 1), Vector2_ == phi^{-1}(tau + 1)
    for (auto tau = 0; tau < N_ - 1; ++tau) {
        TransportImage(Source_, Vector2_, Scalar1_, Queue_); // Scalar1_ <- I(tau)
        ComputeGradScalarField(Scalar1_, Vector4_, Queue_); // Vector4_ <- nabla(I)

        /*if (Regions_.NT() > 1) { // transport POU
            for (auto region = 0; region < Regions_.NT(); ++region) {
                InitialRegions_.ChangeChannel(region);
                Regions_.ChangeChannel(region);

                compute::copy(
                    InitialRegions_.Begin(),
                    InitialRegions_.End(),
                    Scalar1_.Begin(),
                    Queue_
                );

                TransportImage(Scalar1_, Vector2_, Regions_, Queue_);
            }
        }*/

        TransportMomentum(InitialMomentum_, Vector2_, Scalar1_, Queue_); // Scalar1_ <- P(tau)

        ScalarFieldTimesVectorField(Scalar1_, Vector4_, Vector3_, -1.f, Queue_); // Vector3_ <- -P(tau) x nabla(I)
        Convolution(Vector3_, Vector5_, Vector6_); // Vector3_ <- -K (*) [P(tau) x nabla(I)] = v(tau)

        if (tau == 0) { // init GradientMomentum_
            ScalarProduct(Vector4_, Vector3_, GradientMomentum_, -Alpha, Queue_);
            Energy_ +=
                0.5 * DotProduct(GradientMomentum_, InitialMomentum_, Scalar1_, Queue_);
            std::cout << "Energy of the vector field: " << Energy_ << "\n";
        }

        UpdateDiffeo(Vector3_, Vector1_, DeltaT_, Queue_); // Vector1_ <- phi(tau + 1)

        // (Vector4_ is used by UpdateInvDiffeo as a temporary copy of Vector2_)
        UpdateInvDiffeo(Vector3_, Vector2_, Vector4_, DeltaT_, Queue_); // Vector2_ <- phi(tau + 1)

        DiffeoTimeLine_.ChangeChannel(tau);
        compute::copy(
            Vector1_.Begin(),
            Vector1_.End(),
            DiffeoTimeLine_.Begin(),
            Queue_
        );

        InvDiffeoTimeLine_.ChangeChannel(tau);
        compute::copy(
            Vector2_.Begin(),
            Vector2_.End(),
            InvDiffeoTimeLine_.Begin(),
            Queue_
        );
    }
}

float SimilarityMeasure(const GPUScalarField & I, const GPUScalarField & J,
                       const GPUScalarField & acc, compute::command_queue & queue) {
    using compute::lambda::get;
    using compute::_1;

    compute::for_each(
        compute::make_zip_iterator(
            boost::make_tuple(I.Begin(), J.Begin(), acc.Begin())
        ),
        compute::make_zip_iterator(
            boost::make_tuple(I.End(), J.End(), acc.End())
        ),
        get<2>(_1) = pow(get<0>(_1) - get<1>(_1), 2),
        queue
    );

    auto similarityMeasure = 0.f;
    compute::reduce(
        acc.Begin(),
        acc.End(),
        &similarityMeasure,
        compute::plus<float>(),
        queue
    );

    return 0.5f * similarityMeasure;
}

void GeoShoot::ComputeGradient() {
    TransportImage(Source_, Vector2_, Scalar3_, Queue_); // Scalar3_ <- I(1) (Vector2_ still contains phi^{-1}(1))

    Cost_ += SimilarityMeasure(Scalar3_, Target_, Scalar1_, Queue_);
    std::cout << "Similarity measure: " << Cost_ << "\n";

    Cost_ += Energy_;
    std::cout << "Global cost: " << Cost_ << "\n";

    AddFields(Target_, Scalar3_, Scalar4_, -1.f, Queue_); // Scalar4_ <- J - I(1) = \hat{I}(1)
    TransportMomentum(Scalar4_, Vector1_, Scalar1_, Queue_); // Scalar1_ <- \tilde{I}(1) (Vector1_ still contains phi(1))

    Scalar2_.Fill(0.f, Queue_); // Scalar2_ <- \tilde{P}(1) = 0

    // at iteration tau: Scalar1_ == \tilde{I}(tau), Scalar2_ == \tilde{P}(tau)
    for (auto tau = N_ - 2; tau >= 0; --tau) {
        if (tau != N_ - 2) {
            // at tau = N_ - 2 we have just called Shoot(), so Vector1_ still contains phi(1)
            DiffeoTimeLine_.ChangeChannel(tau);
            compute::copy(
                DiffeoTimeLine_.Begin(),
                DiffeoTimeLine_.End(),
                Vector1_.Begin(),
                Queue_
            );

            // same: Vector2_ still contains phi^{-1}(1)
            InvDiffeoTimeLine_.ChangeChannel(tau);
            compute::copy(
                InvDiffeoTimeLine_.Begin(),
                InvDiffeoTimeLine_.End(),
                Vector2_.Begin(),
                Queue_
            );

            // at tau = N_ - 2, Scalar3_ already contains I(1)
            TransportImage(Source_, Vector2_, Scalar3_, Queue_); // Scalar3_ <- I(tau)
        }

        /*if (Regions_.NT() > 1) { // transport POU
            for (auto region = 0; region < Regions_.NT(); ++region) {
                InitialRegions_.ChangeChannel(region);
                Regions_.ChangeChannel(region);

                compute::copy(
                    InitialRegions_.Begin(),
                    InitialRegions_.End(),
                    Scalar4_.Begin(),
                    Queue_
                );

                TransportImage(Scalar4_, Vector2_, Regions_, Queue_);
            }
        }*/
        
        TransportMomentum(Scalar1_, Vector2_, Scalar4_, Queue_); // Scalar4_ <- \hat{I}(tau)

        TransportImage(Scalar2_, Vector2_, Scalar5_, Queue_); // Scalar5_ <- \hat{P}(tau)

        ComputeGradScalarField(Scalar3_, Vector3_, Queue_); // Vector3_ <- nabla(I)

        TransportMomentum(InitialMomentum_, Vector2_, Scalar3_, Queue_); // Scalar3_ <- P(tau)

        /* compute adjoint velocity */
        ComputeGradScalarField(Scalar5_, Vector2_, Queue_); // Vector2_ <- nabla(\hat{P}) (we don't use phi^{-1} anymore so we can overwrite Vector2_)
        ScalarFieldTimesVectorField(Scalar3_, Vector2_, Vector2_, 1.f, Queue_); // Vector2_ <- P x nabla(\hat{P})
        ScalarFieldTimesVectorField(Scalar4_, Vector3_, Vector4_, 1.f, Queue_); // Vector4_ <- \hat{I} x nabla(I)
        AddFields(Vector2_, Vector4_, Vector2_, -1.f, Queue_); // Vector2_ <- P x nabla(\hat{P}) - \hat{I} x nabla(I)

        Convolution(Vector2_, Vector5_, Vector6_); // Vector2_ <- \hat{v}

        ScalarFieldTimesVectorField(Scalar3_, Vector2_, Vector4_, 1.f, Queue_); // Vector4_ <- P x \hat{v}
        ComputeDivVectorField(Vector4_, Scalar4_, Queue_); // Scalar4_ <- div(P x \hat{v})

        TransportMomentum(Scalar4_, Vector1_, Scalar3_, Queue_); // Scalar3_ <- Jac(phi) div(P x \hat{v}) o phi
        AddFields(Scalar1_, Scalar3_, Scalar1_, DeltaT_, Queue_); // Scalar1_ <- \tilde{I}(tau - 1)

        ScalarProduct(Vector3_, Vector2_, Scalar3_, 1.f, Queue_); // Scalar3_ <- nabla(I) . \hat{v}
        TransportImage(Scalar3_, Vector1_, Scalar4_, Queue_); // Scalar4_ <- [nabla(I) . \hat{v}] o phi
        AddFields(Scalar2_, Scalar4_, Scalar2_, -DeltaT_, Queue_); // Scalar2_ <- \tilde{P}(tau - 1)
    }

    AddFields(GradientMomentum_, Scalar2_, GradientMomentum_, -1.f, Queue_); // update GradientMomentum_
}

void GeoShoot::GradientDescent(int iterationsNumber, float gradientStep) {
    auto optimizedCost = Source_.MaxAbsVal(Queue_) + Target_.MaxAbsVal(Queue_);
    optimizedCost *= optimizedCost * NX_ * NY_ * NZ_;

    auto currentCost = optimizedCost;
    auto localCounter = 0;
    for (auto i = 0; i < iterationsNumber; ++i) {
        std::cout << std::endl;
        std::cout << "\tGradient iteration number " << (i + 1) << "\n";
        Shooting();
        ComputeGradient();

        if (Cost_ < currentCost) {
            if (Cost_ < optimizedCost) {
                std::cout << "Global cost decreasing" << std::endl;
                optimizedCost = Cost_;
                localCounter = 0;
            }
        }

        if (Cost_ > currentCost) {
            std::cout << "Global cost increasing" << std::endl;
            ++localCounter;
            if (localCounter == 2) {
                gradientStep *= 0.8f;
                localCounter = 0;
            }
        }

        /*if (Regions_.NT() > 1) {
            InvDiffeoTimeLine_.ChangeChannel(N_ - 2);
            compute::copy(
                InvDiffeoTimeLine_.Begin(),
                InvDiffeoTimeLine_.End(),
                Vector1_.Begin(),
                Queue_
            );

            for (auto region = 0; region < Regions_.NT(); ++region) {
                InitialRegions_.ChangeChannel(region);
                Regions_.ChangeChannel(region);

                compute::copy(
                    InitialRegions_.Begin(),
                    InitialRegions_.End(),
                    Scalar1_.Begin(),
                    Queue_
                );

                TransportImage(Scalar1_, Vector1_, Regions_, Queue_);
            }
        }*/

        if (i != 0 && Regions_.NT() > 1) {
            GradientMomentum_.ChangeChannel(1);
            compute::copy(GradientMomentum_.Begin(), GradientMomentum_.End(), Scalar1_.Begin(), Queue_);

            GradientMomentum_.ChangeChannel(0);
            AddFields(GradientMomentum_, Scalar1_, GradientMomentum_, 1.f, Queue_);
            compute::transform(
                GradientMomentum_.Begin(),
                GradientMomentum_.End(),
                GradientMomentum_.Begin(),
                compute::_1 / 2.f,
                Queue_
            );
        }

        currentCost = Cost_;
        ComputeGradScalarField(Source_, Vector1_, Queue_);
        ScalarFieldTimesVectorField(GradientMomentum_, Vector1_, Vector1_, -1.f, Queue_);
        Convolution(Vector1_, Vector5_, Vector6_);

        auto temp = MaxUpdate / Vector1_.MaxAbsVal(Queue_);

        AddFields(
            InitialMomentum_,
            GradientMomentum_,
            InitialMomentum_,
            -temp * gradientStep,
            Queue_
        );

        if (Regions_.NT() > 1) {
            compute::copy(GradientMomentum_.Begin(), GradientMomentum_.End(), Scalar1_.Begin(), Queue_);
            GradientMomentum_.ChangeChannel(1);
            compute::copy(Scalar1_.Begin(), Scalar1_.End(), GradientMomentum_.Begin(), Queue_);
            GradientMomentum_.ChangeChannel(0);
        }
    }
}

void GeoShoot::Run(int iterationsNumber) {
    iterationsNumber = std::max(1, iterationsNumber);
    assert(Regions_.NT() == ConvolverConfigs.size());

    if (!Init_) {
        for (auto region = 0; region < Regions_.NT(); ++region) {
            auto & cnf = ConvolverConfigs[region];
            auto & cnv = Convolvers_[region];

            for (auto i = 0; i < 7; ++i) {
                cnf.SigmaXs[i] /= Xmm_;
                cnf.SigmaYs[i] /= Ymm_;
                cnf.SigmaZs[i] /= Zmm_;
            }

            /*if (fabs(cnf.Weights[0]) > 0.01f) {
                auto sum = std::accumulate(cnf.Weights.begin(), cnf.Weights.end(), 0.f);
                for (auto && w : cnf.Weights)
                    w /= sum;
            }*/

            cnv.InitiateConvolver(
                NX_, NY_, NZ_,
                cnf.Weights,
                cnf.SigmaXs,
                cnf.SigmaYs,
                cnf.SigmaZs
            );

            if (region == 0) {
                auto dims = cnv.Dims();
                FFTAccumulator_ = GPUVectorField<2> { dims[0], dims[1], dims[2], 1, GetContext() };
            }

            if (fabs(cnf.Weights[0]) < 0.01f) {
                Regions_.ChangeChannel(region);
                std::cout << "Tuning kernels in region " << region << std::endl;
                ReInitiateConvolver_HomoAppaWeights(cnv, cnf);
            }
        }

        Init_ = true;
    }

    GradientDescent(iterationsNumber, 0.5f);
}

void GeoShoot::Save(std::string path) {
    if (path.back() != '/')
        path += '/';

    auto momentum = ScalarField { NX_, NY_, NZ_ };
    compute::copy(InitialMomentum_.Begin(), InitialMomentum_.End(), momentum.Begin(), Queue_);
    auto momentumPath = path + "Momentum.nii";
    momentum.Write({ momentumPath.c_str() });

    auto final = ScalarField { NX_, NY_, NZ_ };
    auto finalPath = path + "FinalDefSrc.nii";
    InvDiffeoTimeLine_.ChangeChannel(N_ - 2);
    compute::copy(
        InvDiffeoTimeLine_.Begin(),
        InvDiffeoTimeLine_.End(),
        Vector1_.Begin(),
        Queue_
    );
    TransportImage(Source_, Vector1_, Scalar1_, Queue_);
    compute::copy(Scalar1_.Begin(), Scalar1_.End(), final.Begin(), Queue_);
    final.Write({ finalPath.c_str() });

    auto dispPath = path + "InvDiffeo";
    auto dispPathX = dispPath + "X.nii", dispPathY = dispPath + "Y.nii",
         dispPathZ = dispPath + "Z.nii";
    InvDiffeoTimeLine_.Write({
        dispPathX.c_str(),
        dispPathY.c_str(),
        dispPathZ.c_str()
    });

    if (Regions_.NT() > 1) { // transport POU
        for (auto region = 0; region < Regions_.NT(); ++region) {
            InitialRegions_.ChangeChannel(region);
            Regions_.ChangeChannel(region);

            compute::copy(
                InitialRegions_.Begin(),
                InitialRegions_.End(),
                Scalar1_.Begin(),
                Queue_
            );

            TransportImage(Scalar1_, Vector1_, Regions_, Queue_);

            compute::copy(
                Regions_.Begin(),
                Regions_.End(),
                InitialRegions_.Begin(),
                Queue_
            );
        }

        auto p = path + "deformedPOU.nii";
        InitialRegions_.Write({p.c_str()});
    }
}

void GeoShoot::ReInitiateConvolver_HomoAppaWeights(FFTConvolver & cnv,
                                                   ConvolverCnf & cnf) {
    ComputeGradScalarField(Target_, Vector1_, Queue_);

    float realW1;

    auto kernel = GetProgram().create_kernel("maxGrad");
    size_t workDim[3] = { (size_t) NX_, (size_t) NY_, (size_t) NZ_ };
    kernel.set_arg(0, Scalar1_.Buffer());
    kernel.set_arg(1, Vector2_.Buffer());
    kernel.set_arg(2, Dims());

    for (auto i = 0; i < 7; ++i) {
        cnv.ChangeKernel(
            { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
            { cnf.SigmaXs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
            { cnf.SigmaYs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
            { cnf.SigmaZs[i], 1.f, 1.f, 1.f, 1.f, 1.f, 1.f }
        );

        AddFields(Source_, Target_, Scalar1_, -1.f, Queue_);
        ScalarFieldTimesVectorField(Scalar1_, Vector1_, Vector2_, 1.f, Queue_);

        if (Regions_.NT() > 1)
            ScalarFieldTimesVectorField(Regions_, Vector2_, Vector2_, 1.f, Queue_);
        cnv.Convolution(Vector2_, FFTAccumulator_);
        if (Regions_.NT() > 1)
            ScalarFieldTimesVectorField(Regions_, Vector2_, Vector2_, 1.f, Queue_);

        Queue_.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
        auto it = compute::max_element(Scalar1_.Begin(), Scalar1_.End(), Queue_);
        auto maxGrad = it.read(Queue_);

        if (!InitTuning_) {
            cnf.Weights[i] = 100.f;
            RealW1_ = 1.f / maxGrad;
            InitTuning_ = true;
        } else
            cnf.Weights[i] = 100.f / maxGrad / RealW1_;

        std::cout << "sigma" << (i + 1) << " = " << (cnf.SigmaXs[i] * Xmm_)
                  << " / weight" << (i + 1) << " = " << cnf.Weights[i] << "\n";
    }

    std::cout << std::endl;
    cnv.ChangeKernel(cnf.Weights, cnf.SigmaXs, cnf.SigmaYs, cnf.SigmaZs, Regions_.NT() == 1);
}