//
//  VectorField.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 21/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#include <vector>
#include <array>

template<size_t Dim>
class VectorField {
private:
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

    VectorField(int NX, int NY, int NZ, int NT) : NX_ { NX }, NY_ { NY }, NZ_ { NZ }, NT_ { NT },
                                                  NXtY_ { NX * NY }, NXtYtZ_ { NX * NY * NZ },
                                                  NXtYtZtT_ { NX * NY * NZ * NT },
                                                  VecField_(Dim * NXtYtZtT_) { }
public:
    using Iterator = std::vector<float>::iterator;
    using ConstIterator = std::vector<float>::const_iterator;

    VectorField() = default;
    static VectorField CreateVoidField(int NX, int NY, int NZ, int NT, float cst = 0);
    static VectorField Read(const std::array<const char *, Dim> & paths);

    // emplace a vector at point (x, y, z)
    void P(std::array<float, Dim> values, int x, int y, int z = 0, int t = 0) {
        auto index = Index0(x, y, z, t);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZtT_ + index] = values[dir];
    }

    // add a vector at point (x, y, z)
    void Add(std::array<float, Dim> values, int x, int y, int z = 0, int t = 0) {
        auto index = Index0(x, y, z, t);
        for (auto dir = 0; dir < Dim; ++dir)
            VecField_[dir * NXtYtZtT_ + index] += values[dir];
    }

    // get value at point (x, y, z) and at direction Direction
    template<size_t Direction>
    float G(int x, int y, int z = 0, int t = 0) const {
        static_assert(Direction < Dim, "Direction should be less than Dim");
        return VecField_[Index<Direction>(x, y, z, t)];
    }

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

template<size_t Dim>
VectorField<Dim> VectorField<Dim>::CreateVoidField(int NX, int NY, int NZ, int NT, float cst) {
    VectorField field { NX, NY, NZ, NT };
    std::fill(field.Begin(), field.End(), cst);
    memset(&field.Image2World_[0], 16 * sizeof(float), 0);
    field.Image2World_[0][0] = field.Image2World_[1][1] = field.Image2World_[2][2]
        = field.Image2World_[3][3] = 1;
    memcpy(&field.World2Image_[0], &field.Image2World_[0], 16 * sizeof(float));
    return field;
}

using ScalarField = VectorField<1>;

template<>
ScalarField ScalarField::Read(const std::array<const char *, 1> & path);

template<>
void ScalarField::Write(const std::array<const char *, 1> & path) const;

template<size_t Dim>
VectorField<Dim> VectorField<Dim>::Read(const std::array<const char *, Dim> & paths) {
    std::array<ScalarField, Dim> directions;
    for (auto dim = 0; dim < Dim; ++dim) {
        directions[dim] = ScalarField::Read({ paths[0] });
        if (dim != 0) {
            auto eq_x = directions[dim].NX() == directions[dim - 1].NX();
            auto eq_y = directions[dim].NY() == directions[dim - 1].NY();
            auto eq_z = directions[dim].NZ() == directions[dim - 1].NZ();
            auto eq_t = directions[dim].NT() == directions[dim - 1].NT();
            if (!eq_x || !eq_y || !eq_z || !eq_t)
                throw std::invalid_argument { "directions do not have the same lengths" };
        }
    }

    VectorField<Dim> field {
        directions[0].NX(), directions[0].NY(), directions[0].NZ(), directions[0].NT()
    };

    for (auto dim = 0; dim < Dim; ++dim) {
        std::copy(
            directions[dim].Begin(),
            directions[dim].End(),
            &field.VecField_[dim * field.NXtYtZtT_]
        );
    }

    field.Image2World_ = std::move(directions[0].Image2World_);
    field.World2Image_ = std::move(directions[0].World2Image_);
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
        dir.Write(paths[dim]);
    }
}

#endif
