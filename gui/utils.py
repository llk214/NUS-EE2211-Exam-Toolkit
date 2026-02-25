import numpy as np


def parse_matrix(s):
    """Parse comma-separated rows into a numpy matrix.
    '1 2 3, 4 5 6' -> [[1,2,3],[4,5,6]]
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty input")
    rows = [r.strip() for r in s.split(',') if r.strip()]
    return np.array([list(map(float, r.split())) for r in rows], dtype=float)


def parse_vector(s):
    """Parse space-separated values into a 1D array."""
    s = s.strip()
    if not s:
        raise ValueError("Empty input")
    return np.array(list(map(float, s.split())), dtype=float)
