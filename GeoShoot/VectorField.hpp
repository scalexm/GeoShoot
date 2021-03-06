/*=========================================================================
 
 VectorField.hpp
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

#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>
#include <vector>
#include <array>
#include <type_traits>

template<size_t Row, size_t Col>
using Matrix = std::array<std::array<float, Col>, Row>;

namespace compute = boost::compute;

struct CPUPolicy {
    using Container = std::vector<float>;

    template<class... Args>
    static Container MakeContainer(const compute::context &, Args && ... args) {
        return Container(std::forward<Args>(args)...);
    }
};

struct GPUPolicy {
    using Container = compute::vector<float>;

    template<class... Args>
    static Container MakeContainer(const compute::context & ctx, Args && ... args) {
        return Container(std::forward<Args>(args)..., ctx);
    }
};

template<size_t Dim, class DevicePolicy>
class GenericVectorField {
private:
    template<size_t OtherDim, class OtherDevicePolicy>
    friend class GenericVectorField;

    int NX_ = 0, NY_ = 0, NZ_ = 0;
    int NXtY_ = 0, NXtYtZ_ = 0;
    Matrix<4, 4> Image2World_, World2Image_;

    size_t Channel_ = 0;

    std::vector<typename DevicePolicy::Container> VecField_;

    template<size_t Direction>
    size_t Index(int x, int y, int z) const {
        return Direction * NXtYtZ_ + z * NXtY_ + y * NX_ + x;
    }

public:
    using Iterator = typename DevicePolicy::Container::iterator;
    using ConstIterator = typename DevicePolicy::Container::const_iterator;

    GenericVectorField() = default;

    GenericVectorField(int NX, int NY, int NZ, int NT = 1,
                       const compute::context & ctx = compute::system::default_context())
                       : NX_ { NX }, NY_ { NY }, NZ_ { NZ },
                         NXtY_ { NX * NY }, NXtYtZ_ { NX * NY * NZ },
                         VecField_(NT) {
        for (auto t = 0; t < NT; ++t)
            VecField_[t] = DevicePolicy::MakeContainer(ctx, Dim * NX * NY * NZ);

        memset(&Image2World_[0], 0, 16 * sizeof(float));
        Image2World_[0][0] = Image2World_[1][1] = Image2World_[2][2] = Image2World_[3][3] = 1;
        World2Image_ = Image2World_;
    }

    const Matrix<4, 4> & Image2World() const {
        return Image2World_;
    }

    const Matrix<4, 4> & World2Image() const {
        return World2Image_;
    }

    void ChangeChannel(size_t c) {
        Channel_ = c;
    }

    // emplace a vector at point (x, y, z)
    void P(std::array<float, Dim> values, int x, int y, int z) {
        auto index = Index<0>(x, y, z);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[Channel_][dir * NXtYtZ_ + index] = values[dir];
    }

    // add a vector at point (x, y, z)
    void Add(std::array<float, Dim> values, int x, int y, int z) {
        auto index = Index<0>(x, y, z);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[Channel_][dir * NXtYtZ_ + index] += values[dir];
    }

    // get value at point (x, y, z) and at direction Direction
    template<size_t Direction>
    float G(int x, int y, int z) const {
        static_assert(Direction < Dim, "Direction should be less than Dim");
        return VecField_[Channel_][Index<Direction>(x, y, z)];
    }

    // set all voxels to value given by cst
    void Fill(float cst) {
        static_assert(std::is_same<DevicePolicy, CPUPolicy>::value, "only on CPU");
        std::fill(Begin(), End(), cst);
    }

    void Fill(float cst, compute::command_queue & queue) {
        compute::fill(Begin(), End(), cst, queue);
    }

    float MaxAbsVal() const {
        static_assert(std::is_same<DevicePolicy, CPUPolicy>::value, "only on CPU");
        auto max = 0.f;
        for (auto && x : VecField_) {
            auto xa = fabs(x);
            if (xa > max)
                max = xa;
        }
        return max;
    }

    float MaxAbsVal(compute::command_queue & queue) const {
        auto it = compute::minmax_element(Begin(), End(), compute::less<float>(), queue);
        return std::max(fabs(it.first.read(queue)), fabs(it.second.read(queue)));
    }

    int NX() const {
        return NX_;
    }

    int NY() const {
        return NY_;
    }

    int NZ() const {
        return NZ_;
    }

    int NT() const {
        return VecField_.size();
    }

    compute::int4_ Dims() const {
        return { NX_, NY_, NZ_, 0 };
    }

    Iterator Begin() {
        return VecField_[Channel_].begin();
    }

    ConstIterator Begin() const {
        return VecField_[Channel_].begin();
    }

    Iterator End() {
        return VecField_[Channel_].end();
    }

    ConstIterator End() const {
        return VecField_[Channel_].end();
    }

    typename DevicePolicy::Container & Buffer() {
        return VecField_[Channel_];
    }

    const typename DevicePolicy::Container & Buffer() const {
        return VecField_[Channel_];
    }

    /* not perfect regarding memory efficiency, should be rewritten if needed */
    static GenericVectorField Read(const std::array<const char *, Dim> & paths) {
        static_assert(std::is_same<DevicePolicy, CPUPolicy>::value, "only on CPU");
        GenericVectorField<Dim, DevicePolicy> field;

        for (auto dim = 0; dim < Dim; ++dim) {
            auto dir = GenericVectorField<1, CPUPolicy>::Read({ paths[dim] });

            if (dim != 0) {
                auto eq_x = dir.NX() == field.NX();
                auto eq_y = dir.NY() == field.NY();
                auto eq_z = dir.NZ() == field.NZ();
                auto eq_t = dir.NT() == field.NT();
                if (!eq_x || !eq_y || !eq_z || !eq_t)
                    throw std::invalid_argument { "directions do not have the same dimensions" };
            }

            if (dim == 0) {
                field = GenericVectorField<Dim, DevicePolicy> {
                    dir.NX(),
                    dir.NY(),
                    dir.NZ(),
                    dir.NT()
                };
                field.Image2World_ = std::move(dir.Image2World_);
                field.World2Image_ = std::move(dir.World2Image_);
            }

            for (auto t = 0; t < dir.NT(); ++t) {
                dir.ChangeChannel(t);
                std::copy(dir.Begin(), dir.End(), &field.VecField_[t][dim * field.NXtYtZ_]);
            }
        }

        return field;
    }

    void Write(const std::array<const char *, Dim> & paths) const {
        static_assert(std::is_same<DevicePolicy, CPUPolicy>::value, "only on CPU");
        for (auto dim = 0; dim < Dim; ++dim) {
            GenericVectorField<1, CPUPolicy> dir { NX_, NY_, NZ_, NT() };

            for (auto t = 0; t < NT(); ++t) {
                dir.ChangeChannel(t);
                std::copy(
                    &VecField_[t][dim * NXtYtZ_],
                    &VecField_[t][dim * NXtYtZ_] + NXtYtZ_,
                    dir.Begin()
                );
            }
            dir.Image2World_ = Image2World_;
            dir.Write({ paths[dim] });
        }
    }
};

template<size_t Dim>
using GPUVectorField = GenericVectorField<Dim, GPUPolicy>;

template<size_t Dim>
using VectorField = GenericVectorField<Dim, CPUPolicy>;

using GPUScalarField = GPUVectorField<1>;
using ScalarField = VectorField<1>;

template<>
ScalarField ScalarField::Read(const std::array<const char *, 1> & path);

template<>
void ScalarField::Write(const std::array<const char *, 1> & path) const;

#endif
