import pytest
import pygraphblas as pgb

from project import single_ssp, multi_ssp


def test_incorrect_start_vertex():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        single_ssp(adj_matrix, 2)


def test_incorrect_start_vertices():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        multi_ssp(adj_matrix, [-2])


def test_not_square():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=1, ncols=2)

    with pytest.raises(ValueError):
        multi_ssp(adj_matrix, [0])


@pytest.mark.parametrize(
    "dim, i, j, v, start_vertex, expected",
    [
        (
            3,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0.4, 19.0, 1.0, 3761.9, 3110.0, 5000.7, 1122.0, 1.1, 9999.5],
            2,
            [1122.0, 1.1, 0.0],
        ),
        (5, [0], [2], [9999], 1, [-1, 0, -1, -1, -1]),
        (
            4,
            [0, 0, 0, 2, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [19, 19, 19, 19, 19, 19],
            0,
            [0, 19, 19, -1],
        ),
    ],
)
def test_single_ssp(dim, i, j, v, start_vertex, expected):
    adj_matrix = pgb.Matrix.from_lists(i, j, v, nrows=dim, ncols=dim)

    assert single_ssp(adj_matrix, start_vertex) == expected


@pytest.mark.parametrize(
    "dim, i, j, v, start_vertices, expected",
    [
        (
            3,
            [0, 0, 1, 1, 2, 2],
            [0, 2, 0, 2, 1, 2],
            [0.0, 5.0, 1.0, 3000.0, 0.0, 3871.0],
            [0, 2],
            [(0, [0.0, 5.0, 5.0]), (2, [1.0, 0.0, 0.0])],
        ),
        (
            5,
            [0],
            [1],
            [59],
            [1, 2],
            [(1, [-1, 0, -1, -1, -1]), (2, [-1, -1, 0, -1, -1])],
        ),
        (
            4,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 3, 1, 8, 5, 1, 1, 1, 1],
            [0, 1, 2],
            [(0, [0, 2, 1, -1]), (1, [2, 0, 1, -1]), (2, [1, 1, 0, -1])],
        ),
    ],
)
def test_multi_ssp(dim, i, j, v, start_vertices, expected):
    adj_matrix = pgb.Matrix.from_lists(i, j, v, nrows=dim, ncols=dim)

    assert multi_ssp(adj_matrix, start_vertices) == expected
