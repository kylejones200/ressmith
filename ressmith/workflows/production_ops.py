"""Production operations workflows.

Provides workflows for production allocation, optimization,
and facility constraint handling.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.production_ops import (
    AllocationResult,
    FacilityConstraints,
    allocate_production_optimal,
    allocate_production_proportional,
    apply_facility_constraints,
    calculate_facility_utilization,
    optimize_production_allocation,
)

logger = logging.getLogger(__name__)


def allocate_production(
    well_capacities: dict[str, float],
    total_target: float,
    method: str = "proportional",
    well_priorities: dict[str, float] | None = None,
    well_costs: dict[str, float] | None = None,
    constraints: FacilityConstraints | None = None,
) -> dict[str, Any]:
    """Allocate production across multiple wells.

    Parameters
    ----------
    well_capacities : dict
        Dictionary mapping well_id to production capacity
    total_target : float
        Total target production rate
    method : str
        Allocation method ('proportional', 'optimal') (default: 'proportional')
    well_priorities : dict, optional
        Dictionary mapping well_id to priority
    well_costs : dict, optional
        Dictionary mapping well_id to cost per unit
    constraints : FacilityConstraints, optional
        Facility constraints

    Returns
    -------
    dict
        Dictionary with allocation results:
        - allocations: Dictionary mapping well_id to allocated rate
        - total_allocated: Total allocated production
        - method: Allocation method used
        - facility_utilization: Facility utilization (if constraints provided)

    Examples
    --------
    >>> capacities = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
    >>> result = allocate_production(capacities, total_target=2000)
    >>> print(f"Total allocated: {result['total_allocated']:.0f}")
    """
    logger.info(f"Allocating production: method={method}, target={total_target:.0f}")

    if method == "proportional":
        allocations = allocate_production_proportional(well_capacities, total_target)
        allocation_method = "proportional"
    elif method == "optimal":
        allocations = allocate_production_optimal(
            well_capacities, well_priorities, total_target, constraints
        )
        allocation_method = "optimal"
    else:
        allocations = allocate_production_proportional(well_capacities, total_target)
        allocation_method = "proportional"

    total_allocated = sum(allocations.values())

    facility_utilization = None
    if constraints and constraints.facility_capacity:
        facility_utilization = calculate_facility_utilization(
            allocations, constraints.facility_capacity
        )

    return {
        "allocations": allocations,
        "total_allocated": float(total_allocated),
        "method": allocation_method,
        "facility_utilization": facility_utilization,
    }


def optimize_production(
    well_capacities: dict[str, float],
    total_target: float,
    well_costs: dict[str, float] | None = None,
    constraints: FacilityConstraints | None = None,
) -> dict[str, Any]:
    """Optimize production allocation with constraints.

    Parameters
    ----------
    well_capacities : dict
        Dictionary mapping well_id to production capacity
    total_target : float
        Total target production rate
    well_costs : dict, optional
        Dictionary mapping well_id to cost per unit
    constraints : FacilityConstraints, optional
        Facility constraints

    Returns
    -------
    dict
        Dictionary with optimization results

    Examples
    --------
    >>> capacities = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
    >>> costs = {'well_1': 10, 'well_2': 12, 'well_3': 15}
    >>> constraints = FacilityConstraints(max_total_rate=2000, facility_capacity=2500)
    >>> result = optimize_production(capacities, total_target=2400, well_costs=costs, constraints=constraints)
    >>> print(f"Optimized allocation: {result['allocations']}")
    """
    logger.info("Optimizing production allocation")

    result = optimize_production_allocation(
        well_capacities, well_costs, total_target, constraints
    )

    return {
        "allocations": result.well_allocations,
        "total_allocated": result.total_allocated,
        "method": result.allocation_method,
        "facility_utilization": result.facility_utilization,
    }


def apply_constraints_to_production(
    production_rates: dict[str, float],
    max_total_rate: float | None = None,
    max_well_rate: float | None = None,
    min_well_rate: float | None = None,
    facility_capacity: float | None = None,
) -> dict[str, float]:
    """Apply facility constraints to production rates.

    Parameters
    ----------
    production_rates : dict
        Dictionary mapping well_id to production rate
    max_total_rate : float, optional
        Maximum total production rate
    max_well_rate : float, optional
        Maximum per-well rate
    min_well_rate : float, optional
        Minimum per-well rate
    facility_capacity : float, optional
        Facility processing capacity

    Returns
    -------
    dict
        Dictionary with constrained rates

    Examples
    --------
    >>> rates = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
    >>> constrained = apply_constraints_to_production(rates, max_total_rate=2000)
    >>> print(f"Total constrained rate: {sum(constrained.values()):.0f}")
    """
    logger.info("Applying facility constraints to production")

    constraints = FacilityConstraints(
        max_total_rate=max_total_rate,
        max_well_rate=max_well_rate,
        min_well_rate=min_well_rate,
        facility_capacity=facility_capacity,
    )

    constrained_rates = apply_facility_constraints(production_rates, constraints)

    logger.info(f"Constrained production: {sum(constrained_rates.values()):.0f}")

    return constrained_rates

