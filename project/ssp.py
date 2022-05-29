from typing import List, Tuple

import pygraphblas as pgb

__all__ = ["single_ssp", "multi_ssp"]


def single_ssp(adj_matrix: pgb.Matrix, start_vertex: int) -> List[int]:
    return multi_ssp(adj_matrix, [start_vertex])[0][1]


def multi_ssp(
    adj_matrix: pgb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if any(vertex < 0 or vertex >= adj_matrix.nrows for vertex in start_vertices):
        raise ValueError(
            f"start_vertices[i] must be between 0 and {adj_matrix.nrows - 1}"
        )

    result = pgb.Matrix.sparse(
        typ=adj_matrix.type, nrows=len(start_vertices), ncols=adj_matrix.ncols
    )

    for i, j in enumerate(start_vertices):
        result.assign_scalar(value=0, row_slice=i, col_slice=j)

    for _ in range(adj_matrix.nrows):
        result.mxm(
            other=adj_matrix,
            semiring=adj_matrix.type.min_plus,
            out=result,
            accum=adj_matrix.type.min,
        )

    def _normalize_result(length: int, vertices, distances) -> List[int]:
        normalized_result = length * [-1]

        for n, vertex in enumerate(vertices):
            normalized_result[vertex] = distances[n]

        return normalized_result

    return [
        (vertex, _normalize_result(adj_matrix.nrows, *result[i].to_lists()))
        for i, vertex in enumerate(start_vertices)
    ]
