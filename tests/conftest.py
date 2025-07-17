#!/usr/bin/env python3
import pytest

from elise.config import NetworkConfig, WeightConfig


@pytest.fixture
def default_weight_config():
    return WeightConfig(
        p=0.5,
        q=0.3,
        p0=0.1,
        W_vis_vis=[0.0, 0.5],
        W_vis_lat=[0.0, 0.5],
        W_lat_lat=[0.0, 0.5],
        W_lat_vis=[0.0, 0.5],
        d_den=[5, 15],
        d_som=[5, 15],
    )


@pytest.fixture
def default_network_config():
    return NetworkConfig(
        num_lat=50,
        num_vis=13,
    )
