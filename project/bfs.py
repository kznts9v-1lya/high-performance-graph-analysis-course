from typing import List, Tuple

import pygraphblas as pgb

__all__ = ["single_bfs", "multi_bfs"]


def single_bfs(adj_matrix: pgb.Matrix, start_vertex: int) -> List[int]:
    return multi_bfs(adj_matrix, [start_vertex])[0][1]


def multi_bfs(
    adj_matrix: pgb.Matrix, start_vertices: List[int]
) -> List[Tuple[int, List[int]]]:
    if not adj_matrix.square:
        raise ValueError("adj_matrix must be square")

    if any(vertex < 0 or vertex >= adj_matrix.nrows for vertex in start_vertices):
        raise ValueError(
            f"start_vertices[i] must be between 0 and {adj_matrix.nrows - 1}"
        )

    if adj_matrix.type != pgb.types.BOOL:
        raise ValueError(
            f"adj_matrix actual type is {adj_matrix.type}\nExpected type is BOOL"
        )

    front = pgb.Matrix.sparse(
        typ=pgb.types.BOOL,
        nrows=len(start_vertices),
        ncols=adj_matrix.ncols,
        fill=False,
    )
    prev_front = pgb.Matrix.sparse(
        typ=pgb.types.BOOL,
        nrows=len(start_vertices),
        ncols=adj_matrix.ncols,
        fill=False,
    )
    result = pgb.Matrix.dense(
        typ=pgb.types.INT64, nrows=len(start_vertices), ncols=adj_matrix.ncols, fill=-1
    )

    for i, j in enumerate(start_vertices):
        front.assign_scalar(True, i, j)
        prev_front.assign_scalar(True, i, j)
        result.assign_scalar(0, i, j)

    i_front = 0
    while True:
        i_front += 1
        prev_nvals = prev_front.nvals

        front.mxm(other=adj_matrix, mask=prev_front, out=front, desc=pgb.descriptor.RC)
        prev_front.eadd(
            other=front,
            add_op=front.type.lxor_monoid,
            out=prev_front,
            desc=pgb.descriptor.R,
        )
        result.assign_scalar(i_front, mask=front)

        if prev_front.nvals == prev_nvals:
            break

    return [(vertex, list(result[i].vals)) for i, vertex in enumerate(start_vertices)]
