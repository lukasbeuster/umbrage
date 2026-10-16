"""Representative-date helpers: declination facts for Singapore and the quadrature rules."""
from datetime import date, datetime

import math

import numpy as np
import pytest

from umbrage.dates import (dates_for, gauss_nodes, interp_weights, lobatto_nodes, poly_weights, year_table,
                           zenith_dates)
from umbrage.sun import SGT, declination_eot

SG_LAT = 1.3521


def test_declination_and_eot_known_values():
    d, e = declination_eot(datetime(2026, 6, 21, 12, tzinfo=SGT))
    assert 23.3 < d < 23.5
    d, e = declination_eot(datetime(2026, 11, 3, 12, tzinfo=SGT))
    assert 16.0 < e < 16.7                     # EoT maximum, early November
    d, e = declination_eot(datetime(2026, 2, 11, 12, tzinfo=SGT))
    assert -14.6 < e < -13.8                   # EoT minimum, mid February
    with pytest.raises(ValueError):
        declination_eot(datetime(2026, 1, 1))


def test_zenith_passages_singapore():
    z = zenith_dates(2026, SG_LAT)
    assert len(z) == 2
    assert date(2026, 3, 22) <= z[0] <= date(2026, 3, 26)
    assert date(2026, 9, 17) <= z[1] <= date(2026, 9, 21)


def test_nodes_and_weights():
    dec, w = lobatto_nodes(5, 23.44)
    assert dec[0] == pytest.approx(23.44) and dec[2] == 0.0 and dec[-1] == pytest.approx(-23.44)
    assert w == pytest.approx([0.125, 0.25, 0.25, 0.25, 0.125])
    dec, w = gauss_nodes(3, 23.44)
    assert dec[1] == pytest.approx(0.0, abs=1e-9) and sum(w) == pytest.approx(1.0)


def test_poly_weights_reproduce_year_mean_of_smooth_function():
    t = year_table(2026)
    dd = np.array([r[1] for r in t])
    f = lambda x: 10 + 0.02 * x ** 2 - 0.001 * x ** 3 + 2 * np.cos(np.radians(4 * x))   # noqa: E731
    truth = f(dd).mean()
    amp = float(np.abs(dd).max())
    for rule in (gauss_nodes, lobatto_nodes):
        dec, w_text = rule(5, amp)
        w, A = poly_weights(dec, dd)
        assert np.dot(w, f(np.array(dec))) == pytest.approx(truth, abs=0.01)
        assert A.sum(axis=0) == pytest.approx(np.ones(len(dd)))
        assert (w > 0).all()
        # textbook weights assume a symmetric year; the real one leans north (mean declination ~ +0.4 deg)
        assert abs(np.dot(w_text, f(np.array(dec))) - truth) > 0.1
    assert 0.2 < dd.mean() < 0.6
    p3 = lambda x: 1 + x - 0.01 * x ** 2 + 0.0003 * x ** 3                              # noqa: E731
    dec, _ = gauss_nodes(4, amp)
    w, _ = poly_weights(dec, dd)
    assert np.dot(w, p3(np.array(dec))) == pytest.approx(p3(dd).mean(), rel=1e-9)        # exact below degree n


def test_interp_weights_partition_the_year():
    t = year_table(2026)
    dd = np.array([r[1] for r in t])
    w, A = interp_weights([23.4, 0.0, -23.4], dd)
    assert A.sum(axis=0) == pytest.approx(np.ones(len(dd)))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 0.2 and w[2] > 0.2           # solstices carry weight, declination dwells there
    w2, _ = interp_weights([5.0, 5.0], dd)      # duplicates split equally
    assert w2 == pytest.approx([0.5, 0.5])


def test_dates_for_rising_and_falling():
    t = year_table(2026)
    up, down = dates_for(0.0, t)
    assert up.month == 3 and 18 <= up.day <= 22
    assert down.month == 9 and 21 <= down.day <= 25
    up, down = dates_for(23.44, t)
    assert abs((up - down).days) <= 3
