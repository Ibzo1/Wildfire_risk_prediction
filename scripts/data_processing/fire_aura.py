import numpy as np
from scipy.ndimage import gaussian_filter


def generate_fire_aura(fire_coords, grid_shape, sigma=3.0, land_sea_mask=None):
    """Generate a continuous fire risk field using Gaussian kernels.

    Parameters
    ----------
    fire_coords : Iterable[Tuple[int, int]]
        Iterable of (lat_idx, lon_idx) pairs indicating ignition points on the
        target grid.
    grid_shape : Tuple[int, int]
        Shape of the output risk field (n_lat, n_lon).
    sigma : float, optional
        Standard deviation for the Gaussian kernel controlling the spatial
        spread. Defaults to 3.0.
    land_sea_mask : ndarray, optional
        Binary mask with ``1`` for land and ``0`` for water. If provided, risk
        values over water are forced to zero.

    Returns
    -------
    numpy.ndarray
        Normalised risk field in the range ``[0, 1]`` with water masking
        applied if ``land_sea_mask`` is given.
    """
    risk_field = np.zeros(grid_shape, dtype=float)

    for lat_idx, lon_idx in fire_coords:
        if 0 <= lat_idx < grid_shape[0] and 0 <= lon_idx < grid_shape[1]:
            impulse = np.zeros(grid_shape, dtype=float)
            impulse[lat_idx, lon_idx] = 1.0
            risk_field += gaussian_filter(impulse, sigma=sigma, mode="constant")

    max_val = risk_field.max()
    if max_val > 0:
        risk_field /= max_val

    if land_sea_mask is not None:
        risk_field *= land_sea_mask

    return risk_field

__all__ = ["generate_fire_aura"]
