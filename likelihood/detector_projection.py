# Credit some part of the source code from bilby

import jax.numpy as jnp

from astropy.constants import c,au,G,pc
from astropy.units import year as yr
from astropy.cosmology import WMAP9 as cosmo

Msun = 4.9255e-6
year = (1*yr).cgs.value
Mpc = 1e6*pc.value/c.value
euler_gamma = 0.577215664901532860606512090082
MR_sun = 1.476625061404649406193430731479084713e3
speed_of_light = 299792458.0
##########################################################
# Construction of arms
##########################################################

def construct_arm(latitude, longitude, arm_tilt, arm_azimuth):
    """

     Args:

        latitude: Latitude in radian
        longitude: Longitude in radian
        arm_tilt: Arm tilt in radian
        arm_azimuth: Arm azimuth in radian
   
    """

    e_long = jnp.array([-jnp.sin(longitude), jnp.cos(longitude), 0])
    e_lat = jnp.array([-jnp.sin(latitude) * jnp.cos(longitude),
                      -jnp.sin(latitude) * jnp.sin(longitude), jnp.cos(latitude)])
    e_h = jnp.array([jnp.cos(latitude) * jnp.cos(longitude),
                    jnp.cos(latitude) * jnp.sin(longitude), jnp.sin(latitude)])

    return (jnp.cos(arm_tilt) * jnp.cos(arm_azimuth) * e_long +
            jnp.cos(arm_tilt) * jnp.sin(arm_azimuth) * e_lat +
            jnp.sin(arm_tilt) * e_h)


def detector_tensor(arm1, arm2):
    return 0.5 * (jnp.einsum('i,j->ij', arm1, arm1) - jnp.einsum('i,j->ij', arm2, arm2))

##########################################################
# Construction of detector tensor
##########################################################

def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    
    Args:

        ra:
        dec:
        time: Greenwich Mean Sidereal Time in geocentric frame
        psi:
        mode:
    """

    gmst = jnp.mod(time, 2 * jnp.pi)
    phi = ra - gmst
    theta = jnp.pi / 2 - dec

    u = jnp.array([jnp.cos(phi) * jnp.cos(theta), jnp.cos(theta) * jnp.sin(phi), -jnp.sin(theta)])
    v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)

    if mode.lower() == 'plus':
        return jnp.einsum('i,j->ij', m, m) - jnp.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return jnp.einsum('i,j->ij', m, n) + jnp.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return jnp.einsum('i,j->ij', m, m) + jnp.einsum('i,j->ij', n, n)

    # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
    omega = jnp.cross(m, n)
    if mode.lower() == 'longitudinal':
        return jnp.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return jnp.einsum('i,j->ij', m, omega) + jnp.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return jnp.einsum('i,j->ij', n, omega) + jnp.einsum('i,j->ij', omega, n)
    else:
        raise ValueError("{} not a polarization mode!".format(mode))

def antenna_response(detector_tensor, ra, dec, time, psi, mode):
    polarization_tensor = get_polarization_tensor(ra, dec, time, psi, mode)
    return jnp.einsum('ij,ij->', detector_tensor, polarization_tensor)

def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c

    Parameters
    ==========
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    detector2: array_like
        Cartesian coordinate vector for the second detector in the geocentric frame.
        To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    time: float
        GPS time in the geocentric frame

    Returns
    =======
    float: Time delay between the two detectors in the geocentric frame

    """
    gmst = jnp.mod(time, 2 * jnp.pi)
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    omega = jnp.array([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
    delta_d = detector2 - detector1
    return jnp.dot(omega, delta_d) / speed_of_light

def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    ==========
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    =======
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * jnp.cos(latitude)**2 +
                                   semi_minor_axis**2 * jnp.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * jnp.cos(latitude) * jnp.cos(longitude)
    y_comp = (radius + elevation) * jnp.cos(latitude) * jnp.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * jnp.sin(latitude)
    return jnp.array([x_comp, y_comp, z_comp])


def get_detector_response(frequency, waveform_polarizations, parameters, detector_tensor, detector_vertex):
    """

     Args:

        ra: Right Ascension in radian
        dec:Right Ascension in radian
        time: Greenwich Mean Sidereal Time in geocentric frame
        psi:
        mode:
   
    """
    signal = {}
    for mode in waveform_polarizations.keys():
        det_response = antenna_response(
            detector_tensor,
            parameters['ra'],
            parameters['dec'],
            parameters['t_c'],
            parameters['psi'], mode)

        signal[mode] = waveform_polarizations[mode] * det_response
    signal_ifo = sum(signal.values())

    time_shift = time_delay_geocentric(detector_vertex, jnp.array([0.,0.,0.]),parameters['ra'], parameters['dec'], parameters['t_c'])

    dt = parameters['geocent_time'] - parameters['start_time']
    dt = dt + time_shift # Note that we always assume the start time of the strain to be 0

    signal_ifo = signal_ifo * jnp.exp(-1j * 2 * jnp.pi * dt * frequency)

    return signal_ifo


