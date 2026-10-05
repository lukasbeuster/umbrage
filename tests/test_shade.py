"""Synthetic geometry with known answers. Run: pytest code/tests -q"""
import math
import numpy as np

from umbrage import cast_shadows


def wall_dsm(n=200, wall_row=100, height=20.0):
    z = np.zeros((n, n), dtype=np.float32)
    z[wall_row, :] = height            # an east-west wall, 1 px thick
    return z


def test_wall_shadow_length_sun_from_south():
    z = wall_dsm()
    # sun due south at 45 deg: shadow falls north (towards row 0) for 20 m
    r = cast_shadows(z, res=1.0, azimuth=180.0, elevation=45.0)
    col = r.building[:, 100]
    assert col[100 - 19]
    assert not col[100 - 22], "shadow longer than h / tan(e)"
    assert col[100 - 10]
    assert not col[100 + 5], "nothing should be shaded on the sunny side"


def test_wall_shadow_length_low_sun():
    z = wall_dsm()
    e = 20.0
    L = 20.0 / math.tan(math.radians(e))      # ~55 m
    r = cast_shadows(z, res=1.0, azimuth=180.0, elevation=e)
    col = r.building[:, 100]
    assert col[100 - int(L) + 2]
    assert not col[100 - int(L) - 2]


def test_sun_from_east_shades_west():
    z = np.zeros((100, 100), dtype=np.float32)
    z[:, 60] = 10.0                     # north-south wall
    r = cast_shadows(z, res=1.0, azimuth=90.0, elevation=45.0)
    row = r.building[50]
    assert row[55] and not row[45] and not row[65]


def test_tree_crown_partial_shade_and_trunk_gap():
    z = np.zeros((120, 120), dtype=np.float32)
    ct = np.full_like(z, np.nan)
    cb = np.full_like(z, np.nan)
    ct[55:65, 55:65] = 10.0            # crown from 4 m to 10 m
    cb[55:65, 55:65] = 4.0
    r = cast_shadows(z, res=1.0, azimuth=180.0, elevation=45.0,
                     canopy_top=ct, canopy_base=cb, tau=0.1)
    assert not r.building.any()
    # ground directly north of the crown, 4..10 m away, sees the crown at 4..10 m height
    assert r.vegetation[55 - 6, 60]
    assert math.isclose(float(r.shade[55 - 6, 60]), 0.9, rel_tol=1e-6)
    # far beyond the crown shadow: sunlit
    assert not r.vegetation[55 - 15, 60]
    # under the north edge of the crown the ray climbs through the crown: shaded
    assert r.vegetation[56, 60]
    # under the south edge, the 45 deg ray leaves the footprint below the 4 m crown base:
    # sunlit through the trunk gap, which is the correct geometry
    assert not r.vegetation[63, 60]


def test_flat_ground_no_shadow():
    z = np.zeros((50, 50), dtype=np.float32)
    r = cast_shadows(z, res=1.0, azimuth=123.0, elevation=30.0)
    assert not r.building.any()
