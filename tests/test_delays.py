#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import pytest

from elise.model import DendriticWeights, SomaticWeights


@pytest.mark.parametrize("weight_class", [DendriticWeights, SomaticWeights])
def test_delay_basic(weight_class, default_weight_config, default_network_config):
    rng = np.random.default_rng(42)
    w = weight_class(default_weight_config, rng_d=rng, rng_w=rng)
    _, delays = w(default_network_config.num_vis, default_network_config.num_lat)
    assert len(delays) == 63  # 50 + 13 = 63
    npt.assert_array_less(4, delays)
    npt.assert_array_less(delays, 16)
    assert np.all(np.equal(np.mod(delays, 1), 0))


@pytest.mark.parametrize("weight_class", [DendriticWeights, SomaticWeights])
def test_delay_consistency(weight_class, default_weight_config, default_network_config):
    rng = np.random.default_rng(42)
    w1 = weight_class(default_weight_config, rng_d=rng, rng_w=rng)
    rng = np.random.default_rng(42)
    w2 = weight_class(default_weight_config, rng_d=rng, rng_w=rng)
    _, delays1 = w1(default_network_config.num_vis, default_network_config.num_lat)
    _, delays2 = w2(default_network_config.num_vis, default_network_config.num_lat)
    npt.assert_allclose(delays1, delays2)
