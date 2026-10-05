"""umbrage: fast 2.5D shade from raster surface models, on CPU or GPU."""
from .shade import ShadeResult, cast_shadows
from .sun import SGT, SunPosition, daylight_timesteps, solar_position

__version__ = "0.1.0.dev0"
__all__ = ["cast_shadows", "ShadeResult", "solar_position", "daylight_timesteps", "SunPosition", "SGT"]
