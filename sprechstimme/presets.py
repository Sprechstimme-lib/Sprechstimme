"""
Preset instrument and synthesizer configurations.
Each preset is a dictionary with configuration for wavetype, filters, envelope, and poly.
"""

from functools import partial
from . import waves
from . import filters

# Preset configurations
_PRESETS = {}

def _register_preset(name, wavetype, filt=None, envelope=None, poly=True):
    """Internal helper to register a preset configuration."""
    _PRESETS[name] = {
        "wavetype": wavetype,
        "filters": filt if filt else [],
        "envelope": envelope,
        "poly": poly
    }

# Synthesizer Presets

_register_preset(
    "piano",
    wavetype=waves.triangle,
    filters=[partial(filters.low_pass, cutoff=4000)],
    envelope={"attack": 0.002, "decay": 0.3, "sustain": 0.5, "release": 0.2},
    poly=True
)

_register_preset(
    "electric_piano",
    wavetype=waves.sine,
    filters=[
        partial(filters.band_pass, low=200, high=5000),
        partial(filters.echo, delay=0.1, decay=0.2)
    ],
    envelope={"attack": 0.005, "decay": 0.2, "sustain": 0.6, "release": 0.15},
    poly=True
)

_register_preset(
    "organ",
    wavetype=waves.sine,
    filters=[],
    envelope={"attack": 0.01, "decay": 0.0, "sustain": 1.0, "release": 0.05},
    poly=True
)

_register_preset(
    "synth_lead",
    wavetype=waves.sawtooth,
    filters=[partial(filters.low_pass, cutoff=2000)],
    envelope={"attack": 0.01, "decay": 0.1, "sustain": 0.7, "release": 0.15},
    poly=False
)

_register_preset(
    "synth_pad",
    wavetype=waves.triangle,
    filters=[
        partial(filters.low_pass, cutoff=1500),
        partial(filters.echo, delay=0.3, decay=0.3)
    ],
    envelope={"attack": 0.5, "decay": 0.2, "sustain": 0.8, "release": 1.0},
    poly=True
)

_register_preset(
    "bass",
    wavetype=waves.sine,
    filters=[partial(filters.low_pass, cutoff=300)],
    envelope={"attack": 0.005, "decay": 0.1, "sustain": 0.5, "release": 0.1},
    poly=False
)

_register_preset(
    "sub_bass",
    wavetype=waves.sine,
    filters=[partial(filters.low_pass, cutoff=150)],
    envelope={"attack": 0.01, "decay": 0.05, "sustain": 0.8, "release": 0.05},
    poly=False
)

_register_preset(
    "pluck",
    wavetype=waves.sawtooth,
    filters=[partial(filters.low_pass, cutoff=3000)],
    envelope={"attack": 0.001, "decay": 0.2, "sustain": 0.1, "release": 0.05},
    poly=True
)

_register_preset(
    "brass",
    wavetype=waves.sawtooth,
    filters=[
        partial(filters.band_pass, low=200, high=4000),
        partial(filters.simple_distortion, gain=1.2, threshold=0.9)
    ],
    envelope={"attack": 0.1, "decay": 0.05, "sustain": 0.9, "release": 0.1},
    poly=True
)

_register_preset(
    "strings",
    wavetype=waves.triangle,
    filters=[
        partial(filters.low_pass, cutoff=3000),
        partial(filters.echo, delay=0.05, decay=0.15)
    ],
    envelope={"attack": 0.2, "decay": 0.1, "sustain": 0.9, "release": 0.3},
    poly=True
)

_register_preset(
    "flute",
    wavetype=waves.sine,
    filters=[partial(filters.band_pass, low=500, high=3000)],
    envelope={"attack": 0.05, "decay": 0.05, "sustain": 0.8, "release": 0.1},
    poly=False
)

_register_preset(
    "bell",
    wavetype=waves.sine,
    filters=[
        partial(filters.high_pass, cutoff=800),
        partial(filters.echo, delay=0.15, decay=0.5)
    ],
    envelope={"attack": 0.001, "decay": 0.5, "sustain": 0.2, "release": 0.8},
    poly=True
)

_register_preset(
    "synth_square",
    wavetype=waves.square,
    filters=[partial(filters.low_pass, cutoff=1500)],
    envelope={"attack": 0.01, "decay": 0.1, "sustain": 0.7, "release": 0.1},
    poly=True
)

_register_preset(
    "chip_lead",
    wavetype=waves.square,
    filters=[],
    envelope={"attack": 0.001, "decay": 0.05, "sustain": 0.6, "release": 0.05},
    poly=False
)

_register_preset(
    "chip_bass",
    wavetype=waves.pulse,
    filters=[partial(filters.low_pass, cutoff=400)],
    envelope={"attack": 0.001, "decay": 0.05, "sustain": 0.7, "release": 0.05},
    poly=False
)

_register_preset(
    "noise_snare",
    wavetype=waves.noise,
    filters=[partial(filters.high_pass, cutoff=1000)],
    envelope={"attack": 0.001, "decay": 0.05, "sustain": 0.1, "release": 0.05},
    poly=False
)

_register_preset(
    "kick",
    wavetype=waves.sine,
    filters=[partial(filters.low_pass, cutoff=100)],
    envelope={"attack": 0.001, "decay": 0.15, "sustain": 0.0, "release": 0.05},
    poly=False
)


def get_preset(name):
    """
    Get a preset configuration by name.
    Returns a dict with keys: wavetype, filters, envelope, poly
    """
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(_PRESETS.keys())}")
    return _PRESETS[name].copy()


def list_presets():
    """Return list of all available preset names."""
    return sorted(_PRESETS.keys())


# allow direct attribute access for common presets
class PresetAccessor:
    """Allows accessing presets as attributes, e.g., presets.piano"""
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"No preset '{name}'")
        return get_preset(name)

    def __dir__(self):
        return list_presets()


# Create singleton accessor instance
_accessor = PresetAccessor()

# Export
__all__ = ['get_preset', 'list_presets']


def __getattr__(name):
    """Module-level attribute access for presets."""
    return getattr(_accessor, name)
