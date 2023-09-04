from multiprocessing import Manager, Process

import numpy as np
import tenseal as ts

# always compute on the second split set
#


class Party2v2:
    def __init__(
        self,
        template,
        n_split,
        row_size_s,
        modulus_degree=4096,
        coeff_bit_sizes=[30, 24, 24, 30],
        precision=24,
        n_threads=-1,
    ):
        self.n = template.shape[0]
        self.template = template
        self.template_split = None
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
        self.template_split = [
            t for t in np.hsplit(self.template, self.n_split)
        ]
        self.row_size_s = row_size_s

    """
    Party1 & Party2 share the workload
    """

    def get_template_split_to_p1(self):
        """
        Split and encrypt J
        Returns:
            split & enc version of J
        """
        if self.n_split == 1:
            end = 1
        else:
            end = self.n_split

        return [
            ts.ckks_tensor(self._context, np.tile(x.T, self.row_size_s))
            for x in self.template_split[:end]
        ]

    def get_partial_res_enc(self, s_from_p1, serialized_context_from_p1=None):
        """
        Compute the partial encrypted result
        Parameters
        ----------
        s_from_p1 : list of ts tensor from P1

        Returns
        -------
        result of the computation

        """
        if self.n_split == 1:
            print(self._context)
            return ts.ckks_vector([], self._context)
        template_to_multiply = [
            np.tile(x.T.flatten(), self.row_size_s)
            for x in self.template_split[self.n_split // 2 + self.n_split % 2 :]
        ]

        if self.n_threads == -1:
            return np.sum(
                [
                    s_from_p1[i] * template_to_multiply[i]
                    for i in range(self.n_split // 2)
                ],
                axis=0,
            )
        else:
            return self._get_partial_res_enc_parallel(
                s_from_p1, serialized_context_from_p1, template_to_multiply
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
        self, s_from_p1, serialized_context_from_p1, template_to_multiply
    ):
        """
        Function used to parallelize the matrix multiplication
        Parameters
        ----------
        s_from_p1 : ts tensors list from P1

        Returns
        -------
        Partial encrypted result
        """

        serialized_context_from_p1 = ts.context_from(serialized_context_from_p1)

        proc_array = []
        shared_list = Manager().list()

        for i in range(self.n_split // 2):
            p = Process(
                target=self.__ex_matrix_mult,
                args=(s_from_p1[i], template_to_multiply[i], shared_list),
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
                ts.ckks_vector_from(context=serialized_context_from_p1, data=ser_ts)
            )

        return sum(part_res)

    def dec_partial_res_from_p1(self, res_from_p1):
        """
        Decrypted the result got from P1
        Parameters
        ----------
        res_from_p1 : enc result from P1

        Returns
        -------
        Cleartext result to P1
        """
        if self.n_threads != -1:
            res_from_p1.link_context(self._context)

        return np.sum(
            np.split(np.array(res_from_p1.decrypt()), self.row_size_s), axis=1
        )

    def get_serialized_context(self):
        _serialized_context = self._context.serialize(
            save_public_key=False,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        return _serialized_context