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

    for _ in range(2):
        result = adj_matrix.mxm(
            other=result, cast=pgb.types.INT64, accum=pgb.types.INT64.PLUS
        )

    result = result.diag().reduce_vector() / 2

    def _normalize_result(length: int, vertices, triangles) -> List[int]:
        normalized_result = length * [0]

        for n, vertex in enumerate(vertices):
            normalized_result[vertex] = triangles[n]

        return normalized_result

    return _normalize_result(adj_matrix.nrows, *result.to_lists())
