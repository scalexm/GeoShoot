//
//  VectorField.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 21/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#include "GPUVectorField.hpp"
#include <vector>
#include <array>

template<size_t Dim>
class VectorField {
private:
    template<size_t OtherDim>
    friend class VectorField;

    int NX_ = 0, NY_ = 0, NZ_ = 0, NT_ = 0;
    int NXtY_ = 0, NXtYtZ_ = 0, NXtYtZtT_ = 0;
    std::array<std::array<float, 4>, 4> Image2World_, World2Image_;

    std::vector<float> VecField_;

    size_t Index0(int x, int y, int z, int t) const {
        return t * NXtYtZ_ + z * NXtY_ + y * NX_ + x;
    }

    template<size_t Direction>
    size_t Index(int x, int y, int z, int t) const {
        return Direction * NXtYtZtT_ + Index0(x, y, z, t);
    }

public:
    using Iterator = std::vector<float>::iterator;
    using ConstIterator = std::vector<float>::const_iterator;

    VectorField() = default;

    VectorField(int NX, int NY, int NZ, int NT) : NX_ { NX }, NY_ { NY }, NZ_ { NZ }, NT_ { NT },
                                                  NXtY_ { NX * NY }, NXtYtZ_ { NX * NY * NZ },
                                                  NXtYtZtT_ { NX * NY * NZ * NT },
                                                  VecField_(Dim * NXtYtZtT_) {
        memset(&Image2World_[0], 0, 16 * sizeof(float));
        Image2World_[0][0] = Image2World_[1][1] = Image2World_[2][2] = Image2World_[3][3] = 1;
        World2Image_ = Image2World_;
    }

    static VectorField Read(const std::array<const char *, Dim> & paths);

    size_t FlatSize() const {
        return VecField_.size();
    }

    // emplace a vector at point (x, y, z)
    void P(std::array<float, Dim> values, int x, int y, int z, int t = 0) {
        auto index = Index0(x, y, z, t);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZtT_ + index] = values[dir];
    }

    // add a vector at point (x, y, z)
    void Add(std::array<float, Dim> values, int x, int y, int z, int t = 0) {
        auto index = Index0(x, y, z, t);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZtT_ + index] += values[dir];
    }

    // get value at point (x, y, z) and at direction Direction
    template<size_t Direction>
    float G(int x, int y, int z, int t = 0) const {
        static_assert(Direction < Dim, "Direction should be less than Dim");
        return VecField_[Index<Direction>(x, y, z, t)];
    }

    // set all voxels to value given by cst
    float Fill(float cst, int t = 0) {
        for (auto dir = 0; dir < Dim; ++dir) {
            for (auto index = 0; index < NXtYtZ_; ++index)
                VecField_[dir * NXtYtZtT_ + t * NXtYtZ_ + index] = cst;
        }
    }

    int NX() const { return NX_; }
    int NY() const { return NY_; }
    int NZ() const { return NZ_; }
    int NT() const { return NT_; }

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

    void Write(const std::array<const char *, Dim> & paths) const;
};

using ScalarField = VectorField<1>;

template<>
ScalarField ScalarField::Read(const std::array<const char *, 1> & path);

template<>
void ScalarField::Write(const std::array<const char *, 1> & path) const;

/* not perfect regarding memory efficiency, should be rewritten if needed */
template<size_t Dim>
VectorField<Dim> VectorField<Dim>::Read(const std::array<const char *, Dim> & paths) {
    VectorField<Dim> field;

    for (auto dim = 0; dim < Dim; ++dim) {
        auto dir = ScalarField::Read({ paths[dim] });

        if (dim != 0) {
            auto eq_x = dir.NX() == field.NX();
            auto eq_y = dir.NY() == field.NY();
            auto eq_z = dir.NZ() == field.NZ();
            auto eq_t = dir.NT() == field.NT();
            if (!eq_x || !eq_y || !eq_z || !eq_t)
                throw std::invalid_argument { "directions do not have the same lengths" };
        }

        if (dim == 0) {
            field = VectorField<Dim> { dir.NX(), dir.NY(), dir.NZ(), dir.NT() };
            field.Image2World_ = std::move(dir.Image2World_);
            field.World2Image_ = std::move(dir.World2Image_);
        }

        std::copy(dir.Begin(), dir.End(), &field.VecField_[dim * field.NXtYtZtT_]);
    }

    return field;
}

template<size_t Dim>
void VectorField<Dim>::Write(const std::array<const char *, Dim> & paths) const {
    for (auto dim = 0; dim < Dim; ++dim) {
        ScalarField dir { NX_, NY_, NZ_, NT_ };
        std::copy(
            &VecField_[dim * NXtYtZtT_],
            &VecField_[dim * NXtYtZtT_] + NXtYtZtT_,
            dir.Begin()
        );
        dir.Image2World_ = Image2World_;
        dir.Write({ paths[dim] });
    }
}

template<size_t Dim>
inline GPUVectorField<Dim> CopyOnDevice(const VectorField<Dim> & field,
                                        compute::command_queue & queue) {
    GPUVectorField<Dim> deviceField {
        field.NX(),
        field.NY(),
        field.NZ(),
        field.NT(),
        queue.get_context()
    };
    compute::copy(field.Begin(), field.End(), deviceField.field.begin(), queue);
    return deviceField;
}

template<size_t Dim>
inline VectorField<Dim> CopyOnHost(const GPUVectorField<Dim> & field,
                                   compute::command_queue & queue) {
    VectorField<Dim> hostField {
        field.NX,
        field.NY,
        field.NZ,
        field.NT,
    };
    compute::copy(field.field.begin(), field.field.end(), hostField.Begin(), queue);
    return hostField;
}

#endif
