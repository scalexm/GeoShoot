//
//  VectorField.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include "VectorField.hpp"
#include "Matrix.hpp"
#include <nifti1_io.h>
#include <cmath>
#include <iostream>
#include <fstream>

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

namespace {
    template<class DataType>
    void FillFromFile(ScalarField & field, const nifti_1_header & hdr, std::ifstream & file) {
        DataType data;
        for (auto z = 0; z < field.NZ(); ++z) {
            for (auto y = 0; y < field.NY(); ++y) {
                for (auto x = 0; x < field.NX(); ++x) {
                    file.read((char*)&data, sizeof(DataType));
                    field.P({ (float)data * hdr.scl_slope + hdr.scl_inter }, x, y, z);
                }
            }
        }
    }
}

template<>
ScalarField ScalarField::Read(const std::array<const char *, 1> & path) {
    std::ifstream file { path[0], std::ios::binary };
    nifti_1_header hdr;

    if (!file)
        throw std::invalid_argument { "bad file" };

    file.read((char *)&hdr, MIN_HEADER_SIZE);

    ScalarField field { hdr.dim[1], hdr.dim[2], hdr.dim[3] };
    if (hdr.sform_code > 0) {
        std::copy(&hdr.srow_x[0], &hdr.srow_x[0] + 4, field.Image2World_[0].begin());
        std::copy(&hdr.srow_y[0], &hdr.srow_y[0] + 4, field.Image2World_[1].begin());
        std::copy(&hdr.srow_z[0], &hdr.srow_z[0] + 4, field.Image2World_[2].begin());
        field.Image2World_[3][0] = field.Image2World_[3][1] = field.Image2World_[3][2] = 0;
        field.Image2World_[3][3] = 1;
    } else if (hdr.qform_code > 0) {
        auto b = hdr.quatern_b, c = hdr.quatern_c, d = hdr.quatern_d;
        auto a = std::sqrtf(1. - (b * b + c * c + d * d));
        auto qfac = hdr.pixdim[0] == -1. ? -1 : 1;

        field.Image2World_[0][0] = hdr.pixdim[1] * (a * a + b * b - c * c - d * d);
        field.Image2World_[1][0] = hdr.pixdim[1] * (2 * b * c + 2 * a * d);
        field.Image2World_[2][0] = hdr.pixdim[1] * (2 * b * d - 2 * a * c);
        field.Image2World_[3][0] = 0;

        field.Image2World_[0][1] = hdr.pixdim[2] * (2 * b * c - 2 * a * d);
        field.Image2World_[1][1] = hdr.pixdim[2] * (a * a + c * c - b * b - d * d);
        field.Image2World_[2][1] = hdr.pixdim[2] * (2 * c * d + 2 * a * b);
        field.Image2World_[3][1] = 0;

        field.Image2World_[0][2] = qfac * hdr.pixdim[3] * (2 * b * d + 2 * a * c);
        field.Image2World_[1][2] = qfac * hdr.pixdim[3] * (2 * c * d - 2 * a * b);
        field.Image2World_[2][2] = qfac * hdr.pixdim[3] * (a * a + d * d - c * c - b * b);
        field.Image2World_[3][2] = 0;

        field.Image2World_[0][3] = hdr.qoffset_x;
        field.Image2World_[0][3] = hdr.qoffset_y;
        field.Image2World_[0][3] = hdr.qoffset_z;
        field.Image2World_[0][3] = 1;
    } else {
        std::cout << "Orientations of " << path[0] << " were basically estimated..." << std::endl;

        field.Image2World_[0][0] = hdr.pixdim[1];
        field.Image2World_[1][0] = 0;
        field.Image2World_[2][0] = 0;
        field.Image2World_[3][0] = 0;

        field.Image2World_[0][1] = 0;
        field.Image2World_[1][1] = hdr.pixdim[2];
        field.Image2World_[2][1] = 0;
        field.Image2World_[3][1] = 0;

        field.Image2World_[0][2] = 0;
        field.Image2World_[1][2] = 0;
        field.Image2World_[2][2] = hdr.pixdim[3];
        field.Image2World_[3][2] = 0;

        field.Image2World_[0][3] = hdr.qoffset_x;
        field.Image2World_[0][3] = hdr.qoffset_y;
        field.Image2World_[0][3] = hdr.qoffset_z;
        field.Image2World_[0][3] = 1;
    }

    field.World2Image_ = Invert4t4Quaternion(field.Image2World_);

    file.seekg((long)hdr.vox_offset);

    if (hdr.scl_slope == 0) {
        hdr.scl_slope = 1;
        std::cout << "Warning the multiplicatory factor of the grey levels (scl_slope) in "
                  << path[0] << " is equal to 0. We set it to 1 in the opened image."
                  << std::endl;
    }

    switch (hdr.datatype) {
        case 2:
            FillFromFile<unsigned char>(field, hdr, file);
            break;
        case 4:
            FillFromFile<short>(field, hdr, file);
            break;
        case 8:
            FillFromFile<int>(field, hdr, file);
            break;
        case 16:
            FillFromFile<float>(field, hdr, file);
            break;
        case 64:
            FillFromFile<double>(field, hdr, file);
            break;
        case 256:
            FillFromFile<char>(field, hdr, file);
            break;
        case 512:
            FillFromFile<unsigned short>(field, hdr, file);
            break;
        case 768:
            FillFromFile<unsigned int>(field, hdr, file);
            break;
        default:
            throw std::runtime_error { "cannot open an image with grey levels of this type" };
    }

    return field;
}

template<>
void ScalarField::Write(const std::array<const char *, 1> & path) const {
    nifti_1_header hdr;
    nifti1_extender pad = { 0, 0, 0, 0 };
    std::ofstream file { path[0], std::ios::binary };

    if (!file)
        throw std::invalid_argument { "bad file" };

    memset((void *)&hdr, 0, sizeof(hdr));
    hdr.sizeof_hdr = MIN_HEADER_SIZE;
    hdr.dim[0] = 4;
    hdr.dim[1] = NX_;
    hdr.dim[2] = NY_;
    hdr.dim[3] = NZ_;
    hdr.dim[4] = 1;
    hdr.datatype = NIFTI_TYPE_FLOAT32;
    hdr.bitpix = 32;
    hdr.qform_code = 0; // should ideally be set to 1 but I don't set the values of 'quatern_b', 'quatern_c' and 'quatern_d'
    hdr.pixdim[1] = sqrt(
        Image2World_[0][0] * Image2World_[0][0] +
        Image2World_[0][1] * Image2World_[0][1] +
        Image2World_[0][2] * Image2World_[0][2]
    );
    hdr.pixdim[2] = sqrt(
        Image2World_[1][0] * Image2World_[1][0] +
        Image2World_[1][1] * Image2World_[1][1] +
        Image2World_[1][2] * Image2World_[1][2]
    );
    hdr.pixdim[3] = sqrt(
        Image2World_[2][0] * Image2World_[2][0] +
        Image2World_[2][1] * Image2World_[2][1] +
        Image2World_[2][2] * Image2World_[2][2]
    );
    hdr.qoffset_x = Image2World_[0][3];
    hdr.qoffset_y = Image2World_[1][3];
    hdr.qoffset_z = Image2World_[2][3];
    hdr.pixdim[4] = 1.0;
    hdr.sform_code = 1;
    std::copy(Image2World_[0].begin(), Image2World_[0].end(), &hdr.srow_x[0]);
    std::copy(Image2World_[1].begin(), Image2World_[1].end(), &hdr.srow_y[0]);
    std::copy(Image2World_[2].begin(), Image2World_[2].end(), &hdr.srow_z[0]);
    hdr.vox_offset = (float) NII_HEADER_SIZE;
    hdr.scl_inter = 0.0;
    hdr.scl_slope = 1.0;
    hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_SEC;
    strncpy(hdr.magic, "n+1\0", 4);

    file.write((char *)&hdr, MIN_HEADER_SIZE);
    file.write((char *)&pad, 4);
    file.write((char *)&VecField_[0], sizeof(float) * VecField_.size());
}