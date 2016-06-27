//
//  Matrix.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include "Matrix.hpp"

Matrix<4, 4> Invert4t4Quaternion(const Matrix<4, 4> & q1) {
    float r11, r12, r13, r21, r22, r23, r31, r32, r33, v1, v2, v3, deti;
    Matrix<4, 4> q2;

    //algorithm inspired from the one of nifti_io.c

    /*  INPUT MATRIX IS:  */
    r11 = q1[0][0]; r12 = q1[0][1]; r13 = q1[0][2];  /* [ r11 r12 r13 v1 ] */
    r21 = q1[1][0]; r22 = q1[1][1]; r23 = q1[1][2];  /* [ r21 r22 r23 v2 ] */
    r31 = q1[2][0]; r32 = q1[2][1]; r33 = q1[2][2];  /* [ r31 r32 r33 v3 ] */
    v1  = q1[0][3]; v2  = q1[1][3]; v3  = q1[2][3];  /* [  0   0   0   1 ] */

    deti = r11*r22*r33-r11*r32*r23-r21*r12*r33
    +r21*r32*r13+r31*r12*r23-r31*r22*r13;

    if( deti != 0.0 ) deti = 1.0 / deti ;

    q2[0][0] = deti*( r22*r33-r32*r23) ;
    q2[0][1] = deti*(-r12*r33+r32*r13) ;
    q2[0][2] = deti*( r12*r23-r22*r13) ;
    q2[0][3] = deti*(-r12*r23*v3+r12*v2*r33+r22*r13*v3
                    -r22*v1*r33-r32*r13*v2+r32*v1*r23) ;

    q2[1][0] = deti*(-r21*r33+r31*r23) ;
    q2[1][1] = deti*( r11*r33-r31*r13) ;
    q2[1][2] = deti*(-r11*r23+r21*r13) ;
    q2[1][3] = deti*( r11*r23*v3-r11*v2*r33-r21*r13*v3
                    +r21*v1*r33+r31*r13*v2-r31*v1*r23) ;

    q2[2][0] = deti*( r21*r32-r31*r22) ;
    q2[2][1] = deti*(-r11*r32+r31*r12) ;
    q2[2][2] = deti*( r11*r22-r21*r12) ;
    q2[2][3] = deti*(-r11*r22*v3+r11*r32*v2+r21*r12*v3
                    -r21*r32*v1-r31*r12*v2+r31*r22*v1) ;

    q2[3][0] = q2[3][1] = q2[3][2] = 0.0 ;
    q2[3][3] = (deti == 0.0) ? 0.0 : 1.0 ; /* failure flag if deti == 0 */
    return q2;
}

/// Solve the problem: MX=D where D is a known vector, M a tridiagonal matrix and X the unknown vector.
/// Inputs are a,b,c,d,n where M(i,i)=b(i), M(i,i-1)=a(i), M(i,i+1)=c(i), D(i)=d(i), D in R^n and M in R^n*R^n.
/// Output is X where X in R^n.  Warning: will modify c and d! */
void TridiagonalSolveFloat(const float * a, const float * b, float * c, float * d,
                           float *x, int n) {
    int i;
    double id;

    /* Modify the coefficients. */
    c[0] /= b[0];                       /* Division by zero risk. */
    d[0] /= b[0];                       /* Division by zero would imply a singular matrix. */
    for (i = 1; i < n; i++) {
        id = (b[i] - c[i-1] * a[i]);      /* Division by zero risk. */
        c[i] /= id;                       /* Last value calculated is redundant. */
        d[i] = (d[i] - d[i-1] * a[i]) / id;
    }

    /* Now back substitute. */
    x[n - 1] = d[n - 1];
    for (i = n - 2; i >= 0; i--)
        x[i] = d[i] - c[i] * x[i + 1];
}