from multiprocessing import Manager, Process
from typing import List, Optional, Union

import numpy as np
import tenseal as ts

#
# always compute on the first split set


class Party1v2:
    def __init__(
        self,
        s,
        n_split,
        modulus_degree=4096,
        coeff_bit_sizes=[30, 24, 24, 30],
        precision=24,
        n_threads=-1,
    ):
        self.n = s.shape[0]
        self.m = s.shape[1]
        self.s: np.ndarray = s
        self.s_split: Optional[Union[List[np.ndarray], List[ts.CKKSVector]]] = None
        self.part_res = None
        self.n_split = n_split
        self.n_threads = n_threads
        self._context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=modulus_degree,
            coeff_mod_bit_sizes=coeff_bit_sizes,
            n_threads=self.n_threads,
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,
        )
        self._context.global_scale = 2**precision
        self._context.generate_galois_keys()
        self.s_split = [t.flatten() for t in np.hsplit(self.s, self.n_split)]

    """
    Party1 & Party2 share the workload
    """

    def get_s_split_to_p2(self):
        """
        Split and encrypt S
        Returns:
            split & enc version of S
        """
        if self.n_split == 1:
            return [ts.ckks_tensor(self._context, [0])]
        return [
            ts.ckks_tensor(self._context, s)
            for s in self.s_split[self.n_split // 2 + self.n_split % 2 :]
        ]

    def get_partial_res_enc(self, template_from_p2, serialized_context_from_p2):
        """
        Compute the partial encrypted result
        Parameters
        ----------
        serialized_context_from_p2
        template_from_p2 : list of ts tensor from P2

        Returns
        -------
        result of the computation
        """

        if self.n_threads == -1:
            return np.sum(
                [
                    template_from_p2[i] * self.s_split[i]
                    for i in range(self.n_split // 2)
                ],
                axis=0,
            )
        else:
            return self._get_partial_res_enc_parallel(
                template_from_p2, serialized_context_from_p2
            )

    """
    Party1 & Party2 share the workload
    Product parallelized 
    """

    def __ex_matrix_mult(self, array_enc, mat, shared_res):
        """
        target method for parallelization
        Parameters
        ----------
        array_enc : ts tensor
        mat : cleartext matrix
        shared_res : shared list for result

        Returns
        -------

        """

        res = array_enc * mat
        # The ts tensor needs to be serialized to be saved and passed between functions
        shared_res.append(res.serialize())

    def _get_partial_res_enc_parallel(
        self, template_from_p2, serialized_context_from_p2
    ):
        """
        Function used to parallelize the matrix multiplication
        Parameters
        ----------
        template_from_p2 : ts tensors list from P2

        Returns
        -------
        Partial encrypted result
        """

        serialized_context_from_p2 = ts.context_from(serialized_context_from_p2)

        proc_array = []
        shared_list = Manager().list()
        for i in range(self.n_split // 2):
            p = Process(
                target=self.__ex_matrix_mult,
                args=(template_from_p2[i], self.s_split[i], shared_list),
            )
            proc_array.append(p)
        for p in proc_array:
            p.start()

        for p in proc_array:
            p.join()

        part_res = list()
        # Deserialization
        for ser_ts in shared_list:
            part_res.append(
                ts.ckks_vector_from(context=serialized_context_from_p2, data=ser_ts)
            )

        return sum(part_res)

    def dec_partial_res_from_p2(self, res_from_p2):
        """
        Decrypted the result got from P2
        Parameters
        ----------
        res_from_p2 : enc result from P2

        Returns
        -------
        Cleartext result to P2
        """
        if self.n_split == 1:
            return []
        if self.n_threads != -1:
            res_from_p2.link_context(self._context)

        self.part_res = np.sum(
            np.split(np.array(res_from_p2.decrypt()), self.get_row_size_S()), axis=1
        )

    def compute_final_res(self, res_from_p2):
        """
        Compute the final result
        Parameters
        ----------
        res_from_p2 : cleartext result from P2

        Returns
        -------
        final result

        """
        res = list()
        if self.n_split != 1:
            res.append(self.part_res)
        res.append(res_from_p2)
        return sum(res)

    def get_row_size_S(self):
        return self.s.shape[0]

    def get_serialized_context(self):
        _serialized_context = self._context.serialize(
            save_public_key=False,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        return _serialized_context