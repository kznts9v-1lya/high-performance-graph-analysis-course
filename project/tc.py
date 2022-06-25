from typing import List

import pygraphblas as pgb

__all__ = ["triangles_counting"]


def triangles_counting(adj_matrix: pgb.Matrix) -> List[int]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if adj_matrix.type != pgb.types.BOOL:
        raise ValueError(
            f"adj_matrix actual type is {adj_matrix.type}\nExpected type is BOOL"
        )

    result = adj_matrix

    # Make graph undirected
    result = result.union(other=result.transpose())
    # Count paths of length 2 and close them with edges of the original graph
    result = result.mxm(other=result, cast=pgb.types.INT64, mask=result)

    def _normalize_result(length: int, res: pgb.Matrix) -> List[int]:
        normalized_result = length * [0]

        for n in range(length):
            normalized_result[n] = res[n].reduce_int() // 2

        return normalized_result

    return _normalize_result(adj_matrix.nrows, result)
