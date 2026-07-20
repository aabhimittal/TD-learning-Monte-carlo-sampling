import numpy as np

from rlkit.utils import moving_average, ascii_sparkline


def test_moving_average_length_and_values():
    out = moving_average([1, 2, 3, 4], window=2)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, [1.0, 1.5, 2.5, 3.5])


def test_moving_average_window_one_is_identity():
    x = [3.0, 1.0, 4.0]
    np.testing.assert_allclose(moving_average(x, window=1), x)


def test_moving_average_handles_empty():
    assert moving_average([], window=5).size == 0


def test_ascii_sparkline_basic():
    s = ascii_sparkline([1, 2, 3, 4, 5], width=50)
    assert len(s) == 5
    assert s[0] != s[-1]  # rising series -> different first/last glyphs


def test_ascii_sparkline_downsamples():
    s = ascii_sparkline(list(range(1000)), width=40)
    assert len(s) == 40


def test_ascii_sparkline_constant():
    s = ascii_sparkline([2, 2, 2, 2])
    assert len(s) == 4
