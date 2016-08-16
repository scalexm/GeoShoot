/*=========================================================================
 
 OpenCL.cl
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

#define U_X (int4)(1, 0, 0, 0)
#define U_Y (int4)(0, 1, 0, 0)
#define U_Z (int4)(0, 0, 1, 0)
#define DIR_X (int4)(0, 0, 0, 0)
#define DIR_Y (int4)(0, 0, 0, 1)
#define DIR_Z (int4)(0, 0, 0, 2)

#define INDEX(coords, dims) (coords.x + coords.y * dims.x + coords.z * dims.x * dims.y \
    + coords.w * dims.x * dims.y * dims.z)

#define G(field, coords) field[INDEX((coords), dims)]

#define GET_GLOBAL_COORDS() (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0)

float interp(__global const float * field, float4 coords, int4 dims) {
    if (coords.x < 0.f) coords.x = 0.0001f;
    if (coords.x >= dims.x - 1.f) coords.x = dims.x - 1.0001f;
    if (coords.y < 0.f) coords.y = 0.0001f;
    if (coords.y >= dims.y - 1.f) coords.y = dims.y - 1.0001;
    if (coords.z < 0.f) coords.z = 0.0001f;
    if (coords.z >= dims.z - 1.f) coords.z = dims.z - 1.0001f;

    int4 icoords = (int4)(coords.x, coords.y, coords.z, coords.w);
    float xwm = 1.f - (coords.x - icoords.x); float xwp = coords.x - icoords.x;
    float ywm = 1.f - (coords.y - icoords.y); float ywp = coords.y - icoords.y;
    float zwm = 1.f - (coords.z - icoords.z); float zwp = coords.z - icoords.z;

    float interpoGreyLevel;
    if (dims.z == 1) { //2D IMAGE
        float wmm = xwm * ywm;
        float wmp = xwm * ywp;
        float wpm = xwp * ywm;
        float wpp = xwp * ywp;

        interpoGreyLevel = wmm * G(field, icoords);
        interpoGreyLevel += wpm * G(field, icoords + U_X);
        interpoGreyLevel += wmp * G(field, icoords + U_Y);
        interpoGreyLevel += wpp * G(field, icoords + U_X + U_Y);
    } else { //3D IMAGE
        float wmmm = xwm * ywm * zwm, wmmp = xwm * ywm * zwp, wmpm = xwm * ywp * zwm,
              wmpp = xwm * ywp * zwp;
        float wpmm = xwp * ywm * zwm, wpmp = xwp * ywm * zwp, wppm = xwp * ywp * zwm,
              wppp = xwp * ywp * zwp;

        interpoGreyLevel = wmmm * G(field, icoords);
        interpoGreyLevel += wpmm * G(field, icoords + U_X);
        interpoGreyLevel += wmpm * G(field, icoords + U_Y);
        interpoGreyLevel += wppm * G(field, icoords + U_X + U_Y);
        interpoGreyLevel += wmmp * G(field, icoords + U_Z);
        interpoGreyLevel += wpmp * G(field, icoords + U_X + U_Z);
        interpoGreyLevel += wmpp * G(field, icoords + U_Y + U_Z);
        interpoGreyLevel += wppp * G(field, icoords + U_X + U_Y + U_Z);
    }

    return interpoGreyLevel;
}

__kernel void gradient(__global float * grad, __global const float * field, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    int z_cond = (dims.z != 1) && (coords.z == 0 || coords.z == dims.z - 1);
    if (coords.x == 0 || coords.y == 0 || coords.x == dims.x - 1 || coords.y == dims.y - 1
        || z_cond) {

        G(grad, coords) = G(grad, coords + DIR_Y) = G(grad, coords + DIR_Z) = 0.f;
    } else {
        float val = G(field, coords + U_X) - G(field, coords - U_X);
        G(grad, coords) = val / 2.f;

        val = G(field, coords + U_Y) - G(field, coords - U_Y);
        G(grad, (coords + DIR_Y)) = val / 2.f;

        if (dims.z != 1)
            val = G(field, coords + U_Z) - G(field, coords - U_Z);
        else
            val = 0.f;
        G(grad, coords + DIR_Z) = val / 2.f;
    }
}
__kernel void div(__global float * div, __global const float * field, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    int z_cond = (dims.z != 1) && (coords.z == 0 || coords.z == dims.z - 1);
    if (coords.x == 0 || coords.y == 0 || coords.x == dims.x - 1 || coords.y == dims.y - 1
        || z_cond) {

        G(div, coords) = 0.f;
    } else {
        float val = G(field, coords + U_X) - G(field, coords - U_X);
        val += G(field, coords + U_Y + DIR_Y) - G(field, coords - U_Y + DIR_Y);
        if (dims.z != 1)
            val += G(field, coords + U_Z + DIR_Z) - G(field, coords - U_Z + DIR_Z);

        G(div, coords) = val / 2.f;
    }
}

__kernel void transportImage(__global float * dst, __global const float * src,
                             __global const float * diffeo, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();
    G(dst, coords) = interp(
        src,
        (float4)(
            G(diffeo, coords),
            G(diffeo, coords + DIR_Y),
            G(diffeo, coords + DIR_Z),
            0.f
        ),
        dims
    );
}

__kernel void transportMomentum(__global float * dst, __global const float * src,
                                __global const float * diffeo, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    int z_cond = (dims.z != 1) && (coords.z == 0 || coords.z == dims.z - 1);
    if (coords.x == 0 || coords.y == 0 || coords.x == dims.x - 1 || coords.y == dims.y - 1
        || z_cond) {

        G(dst, coords) = G(src, coords);
    } else {
        float d11 = (G(diffeo, coords + U_X) - G(diffeo, coords - U_X)) / 2.f;
        float d12 = (G(diffeo, coords + U_Y) - G(diffeo, coords - U_Y)) / 2.f;
        float d13 = 0.f;
        if (dims.z != 1)
            d13 = (G(diffeo, coords + U_Z) - G(diffeo, coords - U_Z)) / 2.f;
        float xx = G(diffeo, coords);

        coords += (int4)(0, 0, 0, 1);
        float d21 = (G(diffeo, coords + U_X) - G(diffeo, coords - U_X)) / 2.f;
        float d22 = (G(diffeo, coords + U_Y) - G(diffeo, coords - U_Y)) / 2.f;
        float d23 = 0.f;
        if (dims.z != 1)
            d23 = (G(diffeo, coords + U_Z) - G(diffeo, coords - U_Z)) / 2.f;
        float yy = G(diffeo, coords);

        coords += (int4)(0, 0, 0, 1);
        float d31 = 0.f, d32 = 0.f, d33 = 1.f;
        if (dims.z != 1) {
            d31 = (G(diffeo, coords + U_X) - G(diffeo, coords - U_X)) / 2.f;
            d32 = (G(diffeo, coords + U_Y) - G(diffeo, coords - U_Y)) / 2.f;
            d33 = (G(diffeo, coords + U_Z) - G(diffeo, coords - U_Z)) / 2.f;
        }
        float zz = G(diffeo, coords);

        float val = interp(src, (float4)(xx, yy, zz, 0.f), dims);

        // Jacobian
        coords -= (int4)(0, 0, 0, 2);
        G(dst, coords) =
            val*(d11*(d22*d33-d32*d23)-d21*(d12*d33-d32*d13)+d31*(d12*d23-d22*d13));
    }
}

__kernel void updateDiffeo(__global float * diffeo, __global const float * velocity,
                           int4 dims, float deltaT) {
    int4 coords = GET_GLOBAL_COORDS();

    float4 coords_f = (float4)(
        G(diffeo, coords),
        G(diffeo, coords + DIR_Y),
        G(diffeo, coords + DIR_Z),
        0.f
    );

    G(diffeo, coords) += deltaT * interp(velocity, coords_f, dims);
    G(diffeo, coords + DIR_Y) += deltaT * interp(velocity, coords_f + (float4)(0, 0, 0, 1), dims);
    G(diffeo, coords + DIR_Z) += deltaT * interp(velocity, coords_f + (float4)(0, 0, 0, 2), dims);
}

float limiter(float a, float b) {
    if (a * b > 0.f)
        return a < b ? a : b;
    return 0.f;
}

float scheme(__global const float * src, __global int * cfl, float eta,
             int4 coords, int4 offset, int4 dims) {
    if (fabs(eta) > 1)
        *cfl = 1;

    float val = G(src, coords), valm = G(src, coords - offset), valp = G(src, coords + offset);
    float deltaB = val - valm;
    float deltaF = valp - val;

    if (eta >= 0.f) {
        float deltaBB = valm - G(src, coords - 2 * offset);
        return -eta * (deltaB + 0.5f * (1.f - eta)
            * (limiter(deltaB, deltaF) - limiter(deltaBB, deltaB)));
    } else {
        float deltaFF = G(src, coords + 2 * offset) - valp;
        return eta * (-deltaF + 0.5f * (1.f + eta)
            * (limiter(deltaFF, deltaF) - limiter(deltaF, deltaB)));
    }
}

__kernel void updateInvDiffeo(__global float * dst, __global const float * velocity,
                              __global const float * src, __global int * cfl,
                              int4 dims, float deltaT) {
    int4 coords = GET_GLOBAL_COORDS();

    int z_cond = (dims.z == 1) || (coords.z > 1 && coords.z < dims.z - 2);
    if (coords.x > 1 && coords.y > 1 && coords.x < dims.x - 2 && coords.y < dims.y - 2 && z_cond) {
        float ratio = deltaT / 1.f;

        for (int dir = 0; dir < 3; ++dir) {
            float val = 0.f;
            int4 newCoords = coords + dir * (int4)(0, 0, 0, 1);

            float eta = ratio * G(velocity, coords);
            val += scheme(src, cfl, eta, newCoords, U_X, dims);

            eta = ratio * G(velocity, coords + DIR_Y);
            val += scheme(src, cfl, eta, newCoords, U_Y, dims);

            if (dims.z != 1) {
                eta = ratio * G(velocity, coords + DIR_Z);
                val += scheme(src, cfl, eta, newCoords, U_Z, dims);
            }

            G(dst, newCoords) += val;
        }
    }
}

__kernel void stimesv(__global float * dst, __global const float * scalar,
                      __global const float * vec, int4 dims, float factor) {
    int4 coords = GET_GLOBAL_COORDS();

    float fs = factor * G(scalar, coords);
    G(dst, coords) = fs * G(vec, coords);
    G(dst, coords + DIR_Y) = fs * G(vec, coords + DIR_Y);
    G(dst, coords + DIR_Z) = fs * G(vec, coords + DIR_Z);
}

__kernel void scalarProduct(__global float * dst, __global const float * src1,
                            __global const float * src2, int4 dims, float factor) {
    int4 coords = GET_GLOBAL_COORDS();

    G(dst, coords) = factor * G(src1, coords) * G(src2, coords);
    G(dst, coords) += factor * G(src1, coords + DIR_Y) * G(src2, coords + DIR_Y);
    G(dst, coords) += factor * G(src1, coords + DIR_Z) * G(src2, coords + DIR_Z);
}

__kernel void projectImage(__global float * dst, __global const float * src,
                           __global const float * transfo, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();
    float4 coords_f = (float4)(
        coords.x*transfo[0]+coords.y*transfo[1]+coords.z*transfo[2]+transfo[3],
        coords.x*transfo[4]+coords.y*transfo[5]+coords.z*transfo[6]+transfo[7],
        coords.x*transfo[8]+coords.y*transfo[9]+coords.z*transfo[10]+transfo[11],
        0.f
    );

    G(dst, coords) = interp(src, coords_f, dims);
}

__kernel void initDiffeo(__global float * diffeo, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    G(diffeo, coords) = coords.x;
    G(diffeo, coords + DIR_Y) = coords.y;
    G(diffeo, coords + DIR_Z) = coords.z;
}

__kernel void maxGrad(__global float * dst, __global const float * field, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    G(dst, coords) = 2.f * sqrt(
        pow(G(field, coords), 2.f)
        + pow(G(field, coords + DIR_Y), 2.f)
        + pow(G(field, coords + DIR_Z), 2.f)
    );
}

__kernel void gaussian(__global float * data, int4 dims,
                       float sigmaX, float sigmaY, float sigmaZ) {
    int4 coords = GET_GLOBAL_COORDS();
    int4 coords2 = coords;

    if (coords.x >= dims.x / 2)
        coords.x -= dims.x;
    if (coords.y >= dims.y / 2)
        coords.y -= dims.y;
    if (coords.z >= dims.z / 2)
        coords.z -= dims.z;

    G(data, coords2) = exp(
        -coords.x * coords.x / (2. * sigmaX * sigmaX)
        -coords.y * coords.y / (2. * sigmaY * sigmaY)
        -coords.z * coords.z / (2. * sigmaZ * sigmaZ)
    );
}

__kernel void addFFT(__global float * filter, __global float * temp, float coeff) {
    int ind = get_global_id(0);
    filter[2 * ind] += temp[ind] * coeff;
}

__kernel void copyFFT(__global float * out, __global const float * in,
                      int NX_out, int NXtY_out, int NXtYtZ_out,
                      int NX_in, int NXtY_in, int NXtYtZ_in,
                      int dirOut, int dirIn,
                      int strideOut, int strideIn) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int indOut = x + y * NX_out + z * NXtY_out + dirOut * NXtYtZ_out;
    int indIn = x + y * NX_in + z * NXtY_in + dirIn * NXtYtZ_in;
    out[strideOut * indOut] = in[strideIn * indIn];
}

__kernel void filter(__global float * signal, __global const float * filter) {
    int ind = 2 * get_global_id(0);
    float a = signal[ind];
    float b = signal[ind + 1];
    float c = filter[ind];
    float d = filter[ind + 1];

    signal[ind] = a * c - b * d;
    signal[ind + 1] = c * b + a * d;
}