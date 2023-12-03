from time import time
from typing import List

import numpy as np
import syft as sy
import torch

from joint_computations.abstract_jc import AbstractJC

PARTY_1 = 0
PARTY_2 = 1
ELEMENT_SIZE = 8
PRECISION_FRACTIONAL = 3


class SPDZ(AbstractJC):
    def __init__(self, n_parties: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hook = sy.TorchHook(torch)
        self.parties: List[sy.VirtualWorker] = [
            sy.VirtualWorker(hook=hook, id=f"party_{i}") for i in range(n_parties)
        ]

        self.crypto_provider = sy.VirtualWorker(hook=hook, id="crypto_provider")
        self.parties[PARTY_1].log_msgs = True
        self.parties[PARTY_2].log_msgs = True
        self.crypto_provider.log_msgs = True

    def mat_mul(self, cubic_spline: np.ndarray, zero_spline: np.ndarray) -> np.ndarray:
        cubic_spline_torch = torch.tensor(cubic_spline)
        zero_spline_torch = torch.tensor(zero_spline).T
        start = time()
        cubic_spline_torch_ptr = cubic_spline_torch.fix_prec(
            precision_fractional=PRECISION_FRACTIONAL
        ).share(
            self.parties[PARTY_1],
            self.parties[PARTY_2],
            crypto_provider=self.crypto_provider,
        )
        zero_spline_torch_ptr = zero_spline_torch.fix_prec(
            precision_fractional=PRECISION_FRACTIONAL
        ).share(
            self.parties[PARTY_1],
            self.parties[PARTY_2],
            crypto_provider=self.crypto_provider,
        )
        end = time()
        party_1_bytes = SPDZ._count_bytes(self.parties[PARTY_1])
        party_2_bytes = SPDZ._count_bytes(self.parties[PARTY_2])
        crypto_provider_bytes = SPDZ._count_bytes(self.crypto_provider)
        self._party_1_total_megabytes += party_1_bytes + crypto_provider_bytes
        self._party_2_total_megabytes += party_2_bytes + crypto_provider_bytes
        self._party_1_total_time += end - start
        self._party_2_total_time += end - start

        joint_hist_ptr = cubic_spline_torch_ptr @ zero_spline_torch_ptr 
        joint_hist_dec: torch.Tensor = joint_hist_ptr.get().float_precision()

        self.parties[PARTY_1].clear_objects()
        self.parties[PARTY_2].clear_objects()
        # remove hook torch
        self.parties[PARTY_1].msg_history = []
        self.parties[PARTY_2].msg_history = []
        return joint_hist_dec.numpy().astype("double")

    @staticmethod
    def _count_bytes(worker):
        """
        Counts the number of bytes. As messages in PySyft seem to be bytes objects we can use the length to determine
        the number of bytes per message:
        https://en.wikiversity.org/wiki/Python_Concepts/Bytes_objects_and_Bytearrays#bytes_objects
        :param worker: The worker.
        :return: The total bytes for this worker.
        """
        total_bytes = 0
        for msg in worker.msg_history:
            # check if msg is instance of tensor
            if isinstance(msg, sy.messaging.message.ObjectMessage):
                tensor = msg.object
                num_elements = tensor.numel()
                total_bytes += num_elements * ELEMENT_SIZE
        return total_bytes
