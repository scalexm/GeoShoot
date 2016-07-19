#define U_X (1, 0, 0, 0)
#define U_Y (0, 1, 0, 0)
#define U_Z (0, 0, 1, 0)
#define DIR_X (0, 0, 0, 0)
#define DIR_Y (0, 0, 0, 1)
#define DIR_Z (0, 0, 0, 2)

#define INDEX(coords, dims) (coords.x + coords.y * dims.x + coords.z * dims.x * dims.y \
    + coords.w * dims.x * dims.y * dims.z)

#define G(field, coords, dims) field[INDEX(coords, dims)]

#define GET_GLOBAL_COORDS() (get_global_id(0), get_global_id(1), get_global_id(2), 0)

float interp(__global const float * field, float4 coords, int4 dims) {
    if (coords.x < 0.f) coords.x = 0.0001f;
    if (coords.x >= dims.x - 1.f) coords.x = dims.x - 1.0001f;
    if (coords.y < 0.f) coords.y = 0.0001f;
    if (coords.y >= dims.y - 1.f) coords.y = dims.y - 1.0001;
    if (coords.z < 0.f) coords.z = 0.0001f;
    if (coords.z >= dims.z - 1.f) coords.z = dims.z - 1.0001f;

    int4 icoords = (coords.x, coords.y, coords.z, coords.w);
    float xwm = 1.f - (coords.x - icoords.x); float xwp = coords.x - icoords.x;
    float ywm = 1.f - (coords.y - icoords.y); float ywp = coords.y - icoords.y;
    float zwm = 1.f - (coords.z - icoords.z); float zwp = coords.z - icoords.z;

    float interpoGreyLevel;
    if (dims.z == 1) { //2D IMAGE
        float wmm = xwm * ywm;
        float wmp = xwm * ywp;
        float wpm = xwp * ywm;
        float wpp = xwp * ywp;

        interpoGreyLevel = wmm * G(field, icoords, dims);
        interpoGreyLevel += wpm * G(field, (icoords + U_X), dims);
        interpoGreyLevel += wmp * G(field, (icoords + U_Y), dims);
        interpoGreyLevel += wpp * G(field, (icoords + U_X + U_Y), dims);
    } else { //3D IMAGE
        float wmmm = xwm * ywm * zwm, wmmp = xwm * ywm * zwp, wmpm = xwm * ywp * zwm,
              wmpp = xwm * ywp * zwp;
        float wpmm = xwp * ywm * zwm, wpmp = xwp * ywm * zwp, wppm = xwp * ywp * zwm,
              wppp = xwp * ywp * zwp;

        interpoGreyLevel = wmmm * G(field, icoords, dims);
        interpoGreyLevel += wpmm * G(field, (icoords + U_X), dims);
        interpoGreyLevel += wmpm * G(field, (icoords + U_Y), dims);
        interpoGreyLevel += wppm * G(field, (icoords + U_X + U_Y), dims);
        interpoGreyLevel += wmmp * G(field, (icoords + U_Z), dims);
        interpoGreyLevel += wpmp * G(field, (icoords + U_X + U_Z), dims);
        interpoGreyLevel += wmpp * G(field, (icoords + U_Y + U_Z), dims);
        interpoGreyLevel += wppp * G(field, (icoords + U_X + U_Y + U_Z), dims);
    }

    return interpoGreyLevel;
}

__kernel void gradient(__global float * grad, __global const float * field, int4 dims) {
    int4 coords = GET_GLOBAL_COORDS();

    int z_cond = (dims.z != 1) && (coords.z == 0 || coords.z == dims.z - 1);
    if (coords.x == 0 || coords.y == 0 || coords.x == dims.x - 1 || coords.y == dims.y - 1
        || z_cond) {

        G(grad, (coords + DIR_X), dims) = G(grad, (coords + DIR_Y), dims)
            = G(grad, (coords + DIR_Z), dims) = 0.f;
    } else {
        float val = (G(field, (coords + U_X), dims) - G(field, (coords - U_X), dims)) / 2.f;
        G(grad, (coords + DIR_X), dims) = val;

        val = (G(field, (coords + U_Y), dims) - G(field, (coords - U_Y), dims)) / 2.f;
        G(grad, (coords + DIR_Y), dims) = val;

        if (dims.z != 1)
            val = (G(field, (coords + U_Z), dims) - G(field, (coords - U_Z), dims)) / 2.f;
        else
            val = 0.f;
        G(grad, (coords + DIR_Z), dims) = val;
    }
}