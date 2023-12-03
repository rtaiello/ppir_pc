#!/usr/bin/env python3

import numpy as np
from rigid_transform import rigid_transform_3D, rigid_transform_2D
import argparse
from joint_computations.clear import Clear
from joint_computations.spdz import SPDZ
from joint_computations.v1.ckks_v1 import CKKSv1
import pandas as pd

if __name__ == "__main__":
    # add args parser
    parser = argparse.ArgumentParser(description="PPIR")
    parser.add_argument("--ppir", type=str, default="clear", help="PPIR protocol")
    args = parser.parse_args()
    if args.ppir == "clear":
        ppir = Clear()
    elif args.ppir == "spdz":
        ppir = SPDZ()
    elif args.ppir == "ckks_v1":
        ppir = CKKSv1(dim_split_vectors=2,n_threads=-1)
    else:
        Exception("PPIR protocol not supported")
df_result = pd.DataFrame(columns=["RMSE", "Communication cost (MB) Party 1", "Communication cost (MB) Party 2", "Time (s) Party 1", "Time (s) Party 2"])
for i in range(1):
    A = np.loadtxt("data/corpus/corpus_callosum_1_vertices.txt") # A = np.loadtxt("bunny_source.txt")
    B = np.loadtxt("data/corpus/corpus_callosum_2_vertices.txt") # B = np.loadtxt("bunny_target.txt") #
    diff = A.shape[0] - B.shape[0]
    if diff > 0:
        # remove random diff rows from A
        A = np.delete(A, np.random.choice(A.shape[0], diff, replace=False), axis=0)
    if diff < 0:
        # remove random diff rows from B
        B = np.delete(B, np.random.choice(B.shape[0], -diff, replace=False), axis=0)
    # add column of ones to A and B from (23,2) to (23,3)
    A = A.T
    A = A + 10
    B = B.T
    # Recover R and t
    ret_R, ret_t = rigid_transform_2D(A, B, ppir)

    # Compare the recovered R and t with the original
    if isinstance(ppir, Clear):
        B2 = (ret_R@A) + ret_t
        np.savetxt("data/results/clear_b2.txt", B2.T)
    else:
        B2 = np.loadtxt("data/results/clear_b2.txt")
        B2_ppir = (ret_R@A) + ret_t
        B2_ppir = B2_ppir.T
        np.savetxt("data/results/clear_b2_ppir.txt", B2_ppir)
        err = B2 - B2_ppir
        err = err * err
        err = np.sum(err)
        rmse = np.sqrt(err/A.shape[1])
        print("RMSE:", rmse)
        print("Communication cost (MB) Party 1:", ppir.party_1_total_megabytes)
        print("Communication cost (MB) Party 2:", ppir.party_2_total_megabytes)
        print("Time (s) Party 1:", ppir.party_1_total_time)
        print("Time (s) Party 2:", ppir.party_2_total_time)
        df_result.loc[i] = [rmse, ppir.party_1_total_megabytes, ppir.party_2_total_megabytes, ppir.party_1_total_time, ppir.party_2_total_time]
df_result.to_csv(f"data/results/{args.ppir}_results.csv")


import matplotlib.pyplot as plt
label_mask = plt.imread("data/corpus/corpus_callosum_1_slice_97.png")[:,:,0]
label_mask = label_mask.astype(np.uint8)

A = A.T
B = B.T
if args.ppir == "clear":
    label = "Clear"
    color = "red"
    B2 = B2

if args.ppir == "spdz":
    label = "PPIR(MPC)"
    color = "red"
    B2 = B2_ppir
if args.ppir == "ckks_v1":
    label = "PPIR(FHEv1)"
    color = "red"
    B2 = B2_ppir

plt.imshow(np.zeros_like(label_mask), cmap='gray')
q = plt.scatter(A[:,1], A[:,0], s=1, c='red')
# add label top left with Q
z = plt.scatter(B[:,1], B[:,0], s=1, c='blue')
plt.axis('off')
plt.legend([q,z], ["Points Q (Moving)", "Points Z (Template)"],  markerscale=6)
plt.savefig("data/results/result_moving_template.png", bbox_inches='tight')
# clear the plot
plt.clf()
plt.imshow(np.zeros_like(label_mask), cmap='gray')
z = plt.scatter(B[:,1], B[:,0], s=1, c='blue')
# add label top left with Z
q_clear = plt.scatter(B2[:,1], B2[:,0], s=1, c=color)
plt.axis('off')


plt.legend([q_clear,z], [f"Points Q Transformed with {label}", "Points Z (Template)"], markerscale=6)
plt.savefig(f"data/results/result_{label.lower()}.png", bbox_inches='tight')

