#!/usr/bin/python

import numpy as np
from joint_computations.clear import Clear
from joint_computations.spdz import SPDZ
from joint_computations.v1.ckks_v1 import CKKSv1

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B, ppir):
    print(A.shape)
    print(B.shape)
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    #H = Am @ np.transpose(Bm)
    H = ppir.mat_mul(Am, Bm)

    
    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    return R, t


def rigid_transform_2D(A, B, ppir):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 2:
        raise Exception(f"Matrix A is not 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"Matrix B is not 2xN, it is {num_rows}x{num_cols}")

    # Find mean column-wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Ensure centroids are 2x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # Subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # Compute the transformation matrix H
    H = ppir.mat_mul(Am, Bm)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Calculate translation
    t = centroid_B - R @ centroid_A
        
    return R, t

