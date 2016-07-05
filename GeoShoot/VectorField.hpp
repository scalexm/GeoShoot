//
//  VectorField.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 21/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>
#include <vector>
#include <array>
#include <type_traits>

namespace compute = boost::compute;

struct CPUPolicy {
    using Container = std::vector<float>;

    template<class... Args>
    static Container MakeContainer(const compute::context & ctx, Args && ... args) {
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
    std::array<std::array<float, 4>, 4> Image2World_, World2Image_;

    typename DevicePolicy::Container VecField_;

    template<size_t Direction>
    size_t Index(int x, int y, int z) const {
        return Direction * NXtYtZ_ + z * NXtY_ + y * NX_ + x;
    }

public:
    using Iterator = typename DevicePolicy::Container::iterator;
    using ConstIterator = typename DevicePolicy::Container::const_iterator;

    GenericVectorField() = default;

    GenericVectorField(int NX, int NY, int NZ,
                       const compute::context & ctx = compute::system::default_context())
                       : NX_ { NX }, NY_ { NY }, NZ_ { NZ },
                         NXtY_ { NX * NY }, NXtYtZ_ { NX * NY * NZ },
                         VecField_ { DevicePolicy::MakeContainer(ctx, Dim * NX * NY * NZ) } {

        memset(&Image2World_[0], 0, 16 * sizeof(float));
        Image2World_[0][0] = Image2World_[1][1] = Image2World_[2][2] = Image2World_[3][3] = 1;
        World2Image_ = Image2World_;
    }

    size_t FlatSize() const {
        return VecField_.size();
    }

    // emplace a vector at point (x, y, z)
    void P(std::array<float, Dim> values, int x, int y, int z) {
        auto index = Index<0>(x, y, z);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZ_ + index] = values[dir];
    }

    // add a vector at point (x, y, z)
    void Add(std::array<float, Dim> values, int x, int y, int z) {
        auto index = Index<0>(x, y, z);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZ_ + index] += values[dir];
    }

    // get value at point (x, y, z) and at direction Direction
    template<size_t Direction>
    float G(int x, int y, int z) const {
        static_assert(Direction < Dim, "Direction should be less than Dim");
        return VecField_[Index<Direction>(x, y, z)];
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
        auto minmax = compute::minmax_element(Begin(), End(), compute::less<float>(), queue);
        return std::max(fabs(*minmax.first), fabs(*minmax.second));
    }

    int NX() const { return NX_; }
    int NY() const { return NY_; }
    int NZ() const { return NZ_; }

    Iterator Begin() {
        return VecField_.begin();
    }

    ConstIterator Begin() const {
        return VecField_.begin();
    }

    Iterator End() {
        return VecField_.end();
    }

    ConstIterator End() const {
        return VecField_.end();
    }

    typename DevicePolicy::Container & Buffer() {
        return VecField_;
    }

    const typename DevicePolicy::Container & Buffer() const {
        return VecField_;
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
                    throw std::invalid_argument { "directions do not have the same lengths" };
            }

            if (dim == 0) {
                field = GenericVectorField<Dim, DevicePolicy> { dir.NX(), dir.NY(), dir.NZ() };
                field.Image2World_ = std::move(dir.Image2World_);
                field.World2Image_ = std::move(dir.World2Image_);
            }

            std::copy(dir.Begin(), dir.End(), &field.VecField_[dim * field.NXtYtZ_]);
        }

        return field;
    }


    void Write(const std::array<const char *, Dim> & paths) const {
        static_assert(std::is_same<DevicePolicy, CPUPolicy>::value, "only on CPU");
        for (auto dim = 0; dim < Dim; ++dim) {
            GenericVectorField<1, CPUPolicy> dir { NX_, NY_, NZ_ };
            std::copy(
                &VecField_[dim * NXtYtZ_],
                &VecField_[dim * NXtYtZ_] + NXtYtZ_,
                dir.Begin()
            );
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
