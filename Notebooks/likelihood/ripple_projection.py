import jax.numpy as jnp
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from likelihood.detector_projection import *


def get_detector_response_ripple(f_list, params_dict, detector_presets):
    wvf_keys = ["Mc", "eta", "chi1", "chi2", "D", "t_c", "phic", "inclination", "psi"]
    wvf_params = jnp.array([params_dict[key] for key in wvf_keys])
    hp, hc = gen_IMRPhenomD_polar(f_list, wvf_params)
    waveform = {"plus": hp, "cross": hc}

    projection_keys = ["ra", "dec", "t_c", "psi"]
    detector_params = {key: params_dict[key] for key in projection_keys}
    detector_params["geocent_time"] = 0.0
    detector_params["start_time"] = 0.0

    data_dict = {}
    for key in detector_presets.keys():
        detector, detector_vertex = detector_presets[key]
        data_dict[key] = get_detector_response(
            f_list, waveform, detector_params, detector, detector_vertex
        )
    return data_dict