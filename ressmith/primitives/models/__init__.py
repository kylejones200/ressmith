"""Decline model classes wrapping primitive functions.

This package contains all decline curve model classes, organized by type:
- arps: Basic ARPS models (exponential, hyperbolic, harmonic)
- advanced: Advanced models for unconventional reservoirs (Duong, power law, stretched exponential)
- variants: Variant models (linear, switch, fixed terminal)
- segmented: Segmented decline model
"""

from ressmith.primitives.models.arps import ArpsExponentialModel, ArpsHarmonicModel, ArpsHyperbolicModel
from ressmith.primitives.models.advanced import DuongModel, PowerLawDeclineModel, StretchedExponentialModel
from ressmith.primitives.models.segmented import SegmentedDeclineModel
from ressmith.primitives.models.variants import FixedTerminalDeclineModel, HyperbolicToExponentialSwitchModel, LinearDeclineModel

__all__ = [
    # ARPS models
    "ArpsExponentialModel",
    "ArpsHyperbolicModel",
    "ArpsHarmonicModel",
    # Advanced models
    "DuongModel",
    "PowerLawDeclineModel",
    "StretchedExponentialModel",
    # Variant models
    "LinearDeclineModel",
    "HyperbolicToExponentialSwitchModel",
    "FixedTerminalDeclineModel",
    # Segmented model
    "SegmentedDeclineModel",
    # Registry
    "MODEL_REGISTRY",
]

# Model registry for workflows (defined after all model classes are imported)
MODEL_REGISTRY = {
    "arps_exponential": ArpsExponentialModel,
    "arps_hyperbolic": ArpsHyperbolicModel,
    "arps_harmonic": ArpsHarmonicModel,
    "linear_decline": LinearDeclineModel,
    "hyperbolic_to_exponential": HyperbolicToExponentialSwitchModel,
    "power_law": PowerLawDeclineModel,
    "duong": DuongModel,
    "stretched_exponential": StretchedExponentialModel,
    "fixed_terminal_decline": FixedTerminalDeclineModel,
}

