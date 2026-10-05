"""Tile a large grid into windows with a halo, so shadows from outside a tile are seen.

Pure index arithmetic; reading and writing rasters is left to the caller (rasterio windows).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    row: int          # tile index
    col: int
    core: tuple       # (r0, r1, c0, c1) in grid pixels: the part this tile owns
    read: tuple       # (r0, r1, c0, c1): core plus halo, clipped to the grid
    crop: tuple       # (r0, r1, c0, c1): where the core sits inside the read window

    @property
    def read_shape(self):
        r0, r1, c0, c1 = self.read
        return r1 - r0, c1 - c0


def tiles(height: int, width: int, tile_px: int, halo_px: int):
    """Yield Tiles covering a height x width grid. Cores do not overlap; reads do."""
    for i, r0 in enumerate(range(0, height, tile_px)):
        r1 = min(r0 + tile_px, height)
        for j, c0 in enumerate(range(0, width, tile_px)):
            c1 = min(c0 + tile_px, width)
            R0, R1 = max(0, r0 - halo_px), min(height, r1 + halo_px)
            C0, C1 = max(0, c0 - halo_px), min(width, c1 + halo_px)
            yield Tile(i, j, (r0, r1, c0, c1), (R0, R1, C0, C1),
                       (r0 - R0, r1 - R0, c0 - C0, c1 - C0))
