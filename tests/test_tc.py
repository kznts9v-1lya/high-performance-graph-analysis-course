import pytest
import pygraphblas as pgb
from project import triangles_counting


def test_not_square():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=1, ncols=2)

    with pytest.raises(ValueError):
        triangles_counting(adj_matrix)


def test_incorrect_type():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        triangles_counting(adj_matrix)


@pytest.mark.parametrize(
    "dim, i, j, v, expected",
    [
        (
            8,
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [1, 3, 3, 4, 6, 3, 5, 6, 5, 6, 5, 6],
            [True] * 12,
            [1, 3, 2, 4, 1, 1, 3, 0],
        ),
        (
            4,
            [1, 1, 2, 3, 2],
            [1, 1, 2, 2, 3],
            [True] * 5,
            [0, 0, 1, 0],
        ),
        (3, [0, 1, 2, 2, 0], [0, 2, 1, 0, 2], [True] * 5, [1, 0, 0]),
        (5, [3], [3], [False], [0] * 5),
    ],
)
def test_triangle_counting(dim, i, j, v, expected):
    adj_matrix = pgb.Matrix.from_lists(i, j, v, nrows=dim, ncols=dim)

    assert triangles_counting(adj_matrix) == expected
