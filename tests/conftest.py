#!/usr/bin/env python3
import pytest

from elise.utils import dict_to_namespace


@pytest.fixture
def default_neuron_config():
    neuron_params = {
        "C_v": 1.0,
        "C_u": 1.0,
        "E_l": -70.0,
        "E_exc": 0.0,
        "E_inh": -75.0,
        "g_l": 0.1,
        "g_den": 2.0,
        "g_exc_0": 0.3,
        "g_inh_0": 6.0,
        "a": 0.3,
        "b": -58.0,
        "lam": 0.7,
    }

    return dict_to_namespace(neuron_params)


@pytest.fixture
def default_weight_config():
    weight_params = {
        "p": 0.5,
        "q": 0.3,
        "p0": 0.1,
        "W_vis_vis": [0.0, 0.5],
        "W_vis_lat": [0.0, 0.5],
        "W_lat_lat": [0.0, 0.5],
        "W_lat_vis": [0.0, 0.5],
        "d_den": [5, 15],
        "d_som": [5, 15],
        "d_int": 25,
    }

    return dict_to_namespace(weight_params)


@pytest.fixture
def default_network_config():
    network_params = {
        "num_lat": 50,
        "num_vis": 13,
    }

    return dict_to_namespace(network_params)
