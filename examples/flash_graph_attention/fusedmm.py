# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Graph Flash Attention (working in progress)
Related work:
- FusedMM: https://arxiv.org/pdf/2011.06391.pdf
- FlashAttention: https://arxiv.org/pdf/2205.14135.pdf
"""

import argparse

import dgl
import numpy as np
import scipy.sparse as sp
import torch as th
import tvm
import tvm.sparse
import tvm.testing
import tvm.tir as tir
from ogb.nodeproppred import DglNodePropPredDataset
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter
from utils import ell, get_dataset


"""
softmax(Q^T * K) * V

sparse matrix: [m, n] with #nonzeros nnz
"""


@T.prim_func
def fusedmm(
    q: T.handle,
    k: T.handle,
    v: T.handle,
    o: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32")
    F = T.dense_fixed(feat_size, idtype="int32")
    J_ = T.dense_fixed(n, idtype="int32")

    Q = T.match_sparse_buffer(q, [I, F], "float32")
    K = T.match_sparse_buffer(k, [J_, F], "float32")
    V = T.match_sparse_buffer(v, [J_, F], "float32")
    O = T.match_sparse_buffer(o, [I, F], "float32")

    score = T.alloc_sparse_buffer([I, J], "float32")
    temp = T.alloc_sparse_buffer(
        [
            I,
        ],
        "float32",
    )
    temp1 = T.alloc_sparse_buffer(
        [
            I,
        ],
        "float32",
    )
    softmax = T.alloc_sparse_buffer([I, J], "float32")
    # Q^T * K
    with T.sp_iter([I, J, F], "SSR", "sddmm") as [i, j, f]:
        with T.init():
            score[i, j] = T.float32(0)
        score[i, j] += Q[i, f] * K[j, f]

    # softmax
    with T.sp_iter([I], "S", "softmax") as [i]:
        with T.sp_iter([J], "R", "computer_max") as [j]:
            with T.init():
                temp[i] = T.min_value("float32")
            temp[i] = T.max(temp[i], score[i, j])
        with T.sp_iter([J], "R", "sum_of_exp") as [j]:
            with T.init():
                temp1[i] = T.float32(0)
            temp1[i] += T.exp(score[i, j] - temp[i], dtype="float32")
        with T.sp_iter([J], "S", "normalize") as [j]:
            softmax[i, j] = T.exp(score[i, j], dtype="float32") / temp1[i]

    # softmax * V
    with T.sp_iter([I, J, F], "SRS", "spmm") as [i, j, f]:
        with T.init():
            O[i, f] = T.float32(0)
        O[i, f] = O[i, f] + softmax[i, j] * V[j, f]


def bench_fusedmm(g, x, y, feat_size):
    indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    print(f"m {m} n {n} nnz{nnz}")
    print("indices", indices.shape)
    print("indptr", indptr.shape)

    mod = tvm.IRModule.from_expr(fusedmm.with_attr("horizontal_fuse", True))

    params = mod["main"].params
    param_map = {
        params[6]: m,  # m
        params[7]: n,  # n
        params[8]: nnz,  # num_tiles,
        params[9]: feat_size,
    }

    mod["main"] = mod["main"].specialize(param_map)

    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)
    print("---------------------------------------")
    print(mod.script())

    # sddmm_blk_outer = sch.get_block("sddmm0")
    # (i,) = sch.get_loops(spmm_blk_outer)
    # io, ii = sch.split(i, [None, 32])
    # sch.bind(ii, "threadIdx.x")
    # sch.bind(io, "blockIdx.x")
    # print(mod["main"].script())
    # mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    # f = tvm.build(mod["main"], target="cuda")
    # dev_module = f.import_module[0]
    # print(dev_module.get_source())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FusedMM in Sparse-TIR")
    parser.add_argument("--dataset", "-d", type=str, default="cora", help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)
    feat_size = 256
    x = th.rand((g.num_src_nodes(), feat_size))
    y = (x @ x.t()) @ x
    print("x", x.shape, "y", y.shape)
    bench_fusedmm(g, x, y, feat_size)
