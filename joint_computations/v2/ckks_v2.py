#
from time import time
from typing import List, Optional

import hydra
import numpy as np

from joint_computations.abstract_jc import AbstractJC
from joint_computations.v2.party1_v2 import Party1v2
from joint_computations.v2.party2_v2 import Party2v2


class CKKSv2(AbstractJC):
    def __init__(
        self,
        dim_split_vectors: int,
        n_threads=-1,
        modulus_degree=4096,
        coeff_bit_sizes=[30, 24, 24, 30],
        precision=24,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dim_split_vectors = dim_split_vectors
        self.party_1: Optional[Party1v2] = None
        self.party_2: Optional[Party2v2] = None
        self.n_split: Optional[int] = None
        self.n_threads: int = n_threads
        self.first_iteration: bool = True
        self.modulus_degree = modulus_degree
        self.coeff_bit_sizes = coeff_bit_sizes
        self.precision = precision

    def shared_workload_protocol(self):
        """
        Shared workload
        """
        hydra.utils.log.info(
            f"src.joint_computations.ckks.ckks_v3.py - Num Split:{self.n_split}"
        )
        # Party 1 sends the second half split of S to Party 2
        start = time()
        s_split_from_p1_enc = self.party_1.get_s_split_to_p2()
        end = time()
        # Compute time and bandwidth for sending S to Party 2
        self._party_1_total_time += end - start
        s_split_enc_bytes = sum([len(s.serialize()) for s in s_split_from_p1_enc])
        self._party_1_total_megabytes += s_split_enc_bytes
        hydra.utils.log.info(
            f"src.joint_computations.ckks.ckks_v3.py - Party1 Enc Time: "
            f"{end - start} (s) - Comm. : {s_split_enc_bytes / 2 ** 20} (MB)"
        )
        # Party 2 sends the first half split of template to Party 1
        start = time()
        template_split_from_p2_enc = self.party_2.get_template_split_to_p1()
        end = time()
        #  Compute time and bandwidth for sending template to Party 1 only the first iteration
        if self.first_iteration:
            self.first_iteration = False
            template_split_enc_bytes = sum(
                [len(s.serialize()) for s in template_split_from_p2_enc]
            )
            self._party_2_total_time += end - start
            self._party_2_total_megabytes += template_split_enc_bytes
            hydra.utils.log.info(
                f"src.joint_computations.ckks.ckks_v3.py - Party2 Enc Time: "
                f"{end - start} (s) - Comm. : {template_split_enc_bytes / 2 ** 20} (MB)"
            )

        hydra.utils.log.info(
            f"src.joint_computations.ckks.ckks_v3.py - Num Split:{self.n_split}"
        )

        serialized_context_from_p2 = self.party_2.get_serialized_context()
        start = time()
        print(template_split_from_p2_enc)
        result_one_side_p1 = self.party_1.get_partial_res_enc(
            template_from_p2=template_split_from_p2_enc,
            serialized_context_from_p2=serialized_context_from_p2,
        )
        end = time()
        self._party_1_total_time += end - start
        self._party_1_total_megabytes += len(result_one_side_p1.tobytes())
        hydra.utils.log.info(
            f"src.joint_computations.ckks.v3.ckks_v3.py - "
            f"Party 1 sent partial encrypted result to Party 2 - "
            f"Time: {end - start} (s) - Comm.: {len(result_one_side_p1.tobytes()) / 2 ** 20} (MB)"
        )

        serialized_context_from_p1 = self.party_1.get_serialized_context()
        start = time()
        result_one_side_p2 = self.party_2.get_partial_res_enc(
            s_from_p1=s_split_from_p1_enc,
            serialized_context_from_p1=serialized_context_from_p1,
        )
        end = time()
        self._party_2_total_time += end - start
        self._party_2_total_megabytes += len(result_one_side_p2.serialize())
        hydra.utils.log.info(
            f"src.joint_computations.ckks.v3.ckks_v3.py - "
            f"Party 2 sent partial encrypted result to Party 1 - "
            f"Time: {end - start} (s) - Comm.: {len(result_one_side_p2.serialize()) / 2 ** 20} (MB)"
        )

        start = time()
        self.party_1.dec_partial_res_from_p2(result_one_side_p2)
        end = time()
        self._party_1_total_time += end - start

        start = time()
        part_res_dec_p2 = self.party_2.dec_partial_res_from_p1(result_one_side_p1)
        end = time()
        self._party_2_total_time += end - start
        self._party_2_total_megabytes += part_res_dec_p2.nbytes
        hydra.utils.log.info(
            f"src.joint_computations.ckks.v3.ckks_v3.py - "
            f"Party 2 sent partial decrypted result to Party 1 - "
            f"Time: {end - start} (s) - Comm.: {part_res_dec_p2.nbytes / 2 ** 20} (MB)"
        )

        start = time()
        result = self.party_1.compute_final_res(part_res_dec_p2)
        end = time()
        self._party_1_total_time += end - start

        # print("P1,P2: ", self.party_1_total_megabytes, self.party_2_total_megabytes)
        return result.reshape(-1, 1)

    def mat_mul(self, s: np.ndarray, template: np.ndarray) -> np.ndarray:
        print(s.shape)
        print(template.shape)
        self.n_split = len(template) // self.dim_split_vectors
        self.modulus_degree, self.coeff_bit_sizes, self.precision,
        self.party_1 = Party1v2(
            s=s,
            n_split=self.n_split,
            modulus_degree=self.modulus_degree,
            coeff_bit_sizes=self.coeff_bit_sizes,
            precision=self.precision,
            n_threads=self.n_threads,
        )
        self.party_2 = Party2v2(
            template=template,
            n_split=self.n_split,
            row_size_s=self.party_1.get_row_size_S(),
            modulus_degree=self.modulus_degree,
            coeff_bit_sizes=self.coeff_bit_sizes,
            precision=self.precision,
            n_threads=self.n_threads,
        )

        return self.shared_workload_protocol()