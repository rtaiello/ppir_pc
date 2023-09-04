from time import time

import numpy as np

from joint_computations.abstract_jc import AbstractJC


class Clear(AbstractJC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mat_tensor_mul(
        self,
        derivative_cubic_spline: np.ndarray,
        zero_spline: np.ndarray,
    ) -> np.ndarray:
        start = time()
        print(derivative_cubic_spline.shape)
        # joint_hist_grad: np.ndarray = np.sum(
        #     zero_spline.T[:, :, None, None] * derivative_cubic_spline[None, :, :, :],
        #     axis=1,
        # )
        # print(derivative_cubic_spline[550:551])
        joint_hist_grad: np.ndarray = np.einsum('ri,ick->rck', zero_spline.T, derivative_cubic_spline)
        # print("computed by einsum")
        # print(joint_hist_grad)
        end = time()
        self._party_1_time_grad_joint += end - start
        self._party_2_time_grad_joint = self._party_1_time_grad_joint
        return joint_hist_grad

    def mat_mul(self, cubic_spline: np.ndarray, zero_spline: np.ndarray) -> np.ndarray:
        print(cubic_spline.shape)
        start = time()
        joint_hist: np.ndarray = np.matmul(cubic_spline, zero_spline.T)
        end = time()
        self._party_1_time_joint += end - start
        self._party_2_time_joint = self._party_1_time_joint
        return joint_hist

    def __str__(self) -> str:
        return "clear"
