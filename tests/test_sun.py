"""Check the in-house solar position against pvlib (if installed) and known facts."""
from datetime import date, datetime

import pytest

from umbrage.sun import SGT, daylight_timesteps, solar_position

SG_LAT, SG_LON = 1.3521, 103.8198


def test_singapore_solar_noon_is_around_1pm():
    steps = daylight_timesteps(date(2026, 3, 21), SG_LAT, SG_LON, step_minutes=1)
    noon = max(steps, key=lambda s: s.elevation)
    assert noon.when.hour == 13 and noon.when.minute < 15
    assert noon.elevation > 88.0


def test_june_sun_is_north_december_sun_is_south():
    jun = solar_position(datetime(2026, 6, 21, 13, 5, tzinfo=SGT), SG_LAT, SG_LON)
    dec = solar_position(datetime(2026, 12, 21, 12, 55, tzinfo=SGT), SG_LAT, SG_LON)
    assert jun.azimuth < 30 or jun.azimuth > 330      # north
    assert 150 < dec.azimuth < 210                      # south
    assert 66 < jun.elevation < 69 and 64 < dec.elevation < 67


def test_against_pvlib():
    pvlib = pytest.importorskip("pvlib")
    import pandas as pd
    times = pd.date_range("2026-05-15 07:30", "2026-05-15 19:00", freq="30min", tz="Asia/Singapore")
    ref = pvlib.solarposition.get_solarposition(times, SG_LAT, SG_LON)
    for t, row in ref.iterrows():
        sp = solar_position(t.to_pydatetime(), SG_LAT, SG_LON)
        if row["apparent_elevation"] > 2:
            assert abs(sp.elevation - row["apparent_elevation"]) < 0.05
            assert abs(((sp.azimuth - row["azimuth"]) + 180) % 360 - 180) < 0.1
