"""Solar position, dependency-free.

NOAA solar position algorithm (the spreadsheet version, after Meeus). Accurate to
roughly 0.01 deg in elevation for the years we care about, which is far below the
error budget of a 1 m shade raster. Checked against pvlib in tests/test_sun.py.

Conventions used everywhere in this repo:
- azimuth: degrees clockwise from true north (0 = N, 90 = E)
- elevation: degrees above the horizon, refraction-corrected
- timestamps: timezone-aware; Singapore is UTC+8 (SGT), no DST
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

SGT = timezone(timedelta(hours=8), name="SGT")


@dataclass(frozen=True)
class SunPosition:
    when: datetime
    azimuth: float      # deg clockwise from north
    elevation: float    # deg above horizon, refraction-corrected
    zenith: float       # 90 - elevation

    @property
    def is_up(self) -> bool:
        return self.elevation > 0.0


def _julian_day(dt_utc: datetime) -> float:
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def declination_eot(when: datetime) -> tuple[float, float]:
    """Solar declination (deg) and equation of time (minutes) at a timezone-aware instant (NOAA).

    The sun's path across the sky on a given day depends only on the declination (for a fixed site); the
    equation of time shifts that path in clock time."""
    if when.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return _decl_eot(when.astimezone(timezone.utc))


def _decl_eot(t_utc: datetime) -> tuple[float, float]:
    jd = _julian_day(t_utc)
    jc = (jd - 2451545.0) / 36525.0

    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(app_long))))

    y = math.tan(math.radians(obliq / 2.0)) ** 2
    l0r = math.radians(l0)
    eot = 4.0 * math.degrees(
        y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr)
    )  # minutes
    return decl, eot


def solar_position(when: datetime, lat: float, lon: float) -> SunPosition:
    if when.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    t_utc = when.astimezone(timezone.utc)
    decl, eot = _decl_eot(t_utc)

    minutes_utc = t_utc.hour * 60 + t_utc.minute + t_utc.second / 60 + t_utc.microsecond / 6e7
    true_solar_time = (minutes_utc + eot + 4.0 * lon) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    if hour_angle < -180.0:
        hour_angle += 360.0

    latr, declr, har = math.radians(lat), math.radians(decl), math.radians(hour_angle)
    cos_zen = math.sin(latr) * math.sin(declr) + math.cos(latr) * math.cos(declr) * math.cos(har)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.degrees(math.acos(cos_zen))

    # azimuth, clockwise from north
    sin_zen = math.sin(math.radians(zen))
    if abs(sin_zen) < 1e-9:
        az = 180.0
    else:
        cos_az = (math.sin(latr) * cos_zen - math.sin(declr)) / (math.cos(latr) * sin_zen)
        cos_az = max(-1.0, min(1.0, cos_az))
        az_raw = math.degrees(math.acos(cos_az))
        az = (az_raw + 180.0) % 360.0 if hour_angle > 0 else (540.0 - az_raw) % 360.0

    # atmospheric refraction (NOAA)
    elev = 90.0 - zen
    if elev > 85.0:
        refr = 0.0
    elif elev > 5.0:
        te = math.tan(math.radians(elev))
        refr = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    elif elev > -0.575:
        refr = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    else:
        refr = -20.772 / math.tan(math.radians(elev))
    elev += refr / 3600.0

    return SunPosition(when=when, azimuth=az, elevation=elev, zenith=90.0 - elev)


def daylight_timesteps(day: date, lat: float, lon: float, step_minutes: int = 30,
                       min_elevation: float = 0.0, tz: timezone = SGT,
                       start: time = time(6, 0), end: time = time(20, 0)) -> list[SunPosition]:
    """All timesteps on `day` (local time, every `step_minutes`) with the sun above `min_elevation`."""
    out = []
    t = datetime.combine(day, start, tzinfo=tz)
    t_end = datetime.combine(day, end, tzinfo=tz)
    while t <= t_end:
        sp = solar_position(t, lat, lon)
        if sp.elevation > min_elevation:
            out.append(sp)
        t += timedelta(minutes=step_minutes)
    return out
