from time import time

import numpy as np

from joint_computations.abstract_jc import AbstractJC


class Clear(AbstractJC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def mat_mul(self, cubic_spline: np.ndarray, zero_spline: np.ndarray) -> np.ndarray:
        print(cubic_spline.shape)
        start = time()
        joint_hist: np.ndarray = np.matmul(cubic_spline, zero_spline.T)
        end = time()
        self._party_1_total_time += end - start
        self._party_2_total_time = self._party_1_total_time
        return joint_hist

    def __str__(self) -> str:
        return "clear"
