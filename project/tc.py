from typing import List

import pygraphblas as pgb


def triangles_counting(adj_matrix: pgb.Matrix) -> List[int]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if adj_matrix.type != pgb.types.BOOL:
        raise ValueError(
            f"adj_matrix actual type is {adj_matrix.type}\nExpected type is BOOL"
        )

    result = adj_matrix

    result = result.union(other=result.transpose())
    result = result.mxm(other=result, cast=pgb.types.INT64, mask=result)

    def _normalize_result(length: int, res: pgb.Matrix) -> List[int]:
        normalized_result = length * [0]

        for n in range(length):
            normalized_result[n] = res[n].reduce_int() // 2

        return normalized_result

    return _normalize_result(adj_matrix.nrows, result)
