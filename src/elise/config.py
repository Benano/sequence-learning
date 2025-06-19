#!
# /usr/bin/env python3
import tomllib as toml
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Tuple


@dataclass(slots=True)
class NetworkConfig:
    num_lat: int = 50
    num_vis: int = 13


@dataclass(slots=True)
class WeightConfig:
    w_som_seed: int = 42
    w_den_seed: int = 99
    d_som_seed: int = 42
    d_den_seed: int = 99
    p: float = 0.5
    q: float = 0.3
    p0: float = 0.1
    den_spars: float = 0.3
    W_vis_vis: tuple = (0.0, 0.5)
    W_vis_lat: tuple = (0.0, 0.5)
    W_lat_lat: tuple = (0.0, 0.5)
    W_lat_vis: tuple = (0.0, 0.5)
    d_den: Tuple[int, int] = (5, 15)
    d_som: Tuple[int, int] = (5, 15)
    d_int: int = 25


@dataclass(slots=True)
class SimulationConfig:
    input: str = "test"
    dt: float = 0.01
    pattern_dt: float = 0.25
    pattern_duration: float = 100.0
    replay_epochs: int = 10
    training_epochs: int = 10
    training_cycles: int = 1000
    validation_cycles: int = 100
    replay_cycles: int = 100
    validation_interval: int = 20
    eta_out: float = 10e-4
    eta_lat: float = 10e-3


@dataclass(slots=True)
class NeuronConfig:
    C_v: float = 1.0
    C_u: float = 1.0
    E_l: float = -70.0
    E_exc: float = 0.0
    E_inh: float = -75.0
    g_l: float = 0.1
    g_den: float = 2.0
    g_exc_0: float = 0.3
    g_inh_0: float = 6.0
    a: float = 0.3
    b: float = -58.0
    lam: float = 0.6


@dataclass(slots=True)
class TrackingConfig:
    sim_step: int = 2
    epoch_step: int = 1
    vars_train: list = field(default_factory=list)
    vars_epoch: list = field(default_factory=list)
    vars_replay: list = field(default_factory=list)
    vars_val: list = field(default_factory=list)


class Config:
    """
    A class for loading and accessing configuration data from a TOML file.

    Attributes:
        _config (dict): The loaded configuration data.

    Methods:
        get_section(section: str) -> Dict[str, Any]:
    Retrieves a specific section from the configuration.
    """

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            self._config = toml.load(f)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieves a specific section from the configuration.

        Args: section (str): The name of the section to retrieve.

        Returns:
            Dict[str, Any]:
            A dictionary containing the configuration data for the specified section.
        """
        return self._config.get(section, {})


class FullConfig:
    def __init__(self, config_file: str):
        config = Config(config_file)
        self.seed: int = config.get_section("").get("seed", 69)
        self.network_params = self._create_config(
            NetworkConfig, config.get_section("network_params")
        )
        self.weight_params = self._create_config(
            WeightConfig, config.get_section("weight_params")
        )
        self.simulation_params = self._create_config(
            SimulationConfig, config.get_section("simulation_params")
        )
        self.neuron_params = self._create_config(
            NeuronConfig, config.get_section("neuron_params")
        )
        self.tracking_params = self._create_config(
            TrackingConfig, config.get_section("tracking_params")
        )

    def _create_config(self, config_class: Config, config_dict: Dict[str, Any]):
        kwargs = {}
        for f in fields(config_class):
            if f.name in config_dict:
                kwargs[f.name] = config_dict[f.name]
        return config_class(**kwargs)
