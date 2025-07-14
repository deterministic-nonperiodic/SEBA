import numpy as np

from .constants import Omega


def coriolis_parameter(latitude):
    r"""Calculate the coriolis parameter at each point.
    The implementation uses the formula outlined in [Hobbs1977]_ pg.370-371.
    Parameters
    ----------
    :param latitude: array
        Latitude at each point

    returns coriolis parameter
    """
    return 2.0 * Omega * np.sin(np.deg2rad(latitude))
