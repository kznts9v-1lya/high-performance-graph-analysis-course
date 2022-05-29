import pytest
import pygraphblas as pgb
from project import single_bfs, multi_bfs


def test_incorrect_start_vertex():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        single_bfs(adj_matrix, 2)


def test_incorrect_start_vertices():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        multi_bfs(adj_matrix, [-2])


def test_not_square():
    adj_matrix = pgb.Matrix.dense(pgb.BOOL, nrows=1, ncols=2)

    with pytest.raises(ValueError):
        multi_bfs(adj_matrix, [0])


def test_incorrect_type():
    adj_matrix = pgb.Matrix.dense(pgb.INT64, nrows=2, ncols=2)

    with pytest.raises(ValueError):
        multi_bfs(adj_matrix, [0])


@pytest.mark.parametrize(
    "dim, i, j, v, start_vertex, expected",
    [
        (
            5,
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            3,
            [-1, -1, -1, 0, 1],
        ),
        (
            5,
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [False, True, True, True, True],
            0,
            [0, 2, 1, -1, -1],
        ),
        (
            5,
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            0,
            [0, 1, 1, -1, -1],
        ),
        (
            5,
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            0,
            [0, 1, 2, 3, 4],
        ),
        (5, [0], [1], [False], 0, [0, -1, -1, -1, -1]),
    ],
)
def test_single_bfs(dim, i, j, v, start_vertex, expected):
    adj_matrix = pgb.Matrix.from_lists(i, j, v, nrows=dim, ncols=dim)

    assert single_bfs(adj_matrix, start_vertex) == expected


@pytest.mark.parametrize(
    "dim, i, j, v, start_vertices, expected",
    [
        (
            5,
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            [0, 3],
            [(0, [0, 1, 1, -1, -1]), (3, [-1, -1, -1, 0, 1])],
        ),
        (
            5,
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            [True, True, True, True, True],
            [0],
            [(0, [0, 1, 1, -1, -1])],
        ),
        (
            5,
            [0],
            [1],
            [False],
            [0, 2],
            [(0, [0, -1, -1, -1, -1]), (2, [-1, -1, 0, -1, -1])],
        ),
        (
            5,
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [True, True, True, True],
            [0, 1, 2, 3, 4],
            [
                (0, [0, 1, 2, 3, 4]),
                (1, [-1, 0, 1, 2, 3]),
                (2, [-1, -1, 0, 1, 2]),
                (3, [-1, -1, -1, 0, 1]),
                (4, [-1, -1, -1, -1, 0]),
            ],
        ),
    ],
)
def test_multi_bfs(dim, i, j, v, start_vertices, expected):
    adj_matrix = pgb.Matrix.from_lists(i, j, v, nrows=dim, ncols=dim)

    assert multi_bfs(adj_matrix, start_vertices) == expected
