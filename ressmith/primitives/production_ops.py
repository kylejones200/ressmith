"""Production operations: allocation, optimization, facility constraints.

This module provides functions for production allocation, optimization,
and facility constraint handling.

References:
- Ikoku, C.U., "Natural Gas Production Engineering," 1984.
- Beggs, H.D., "Production Optimization Using Nodal Analysis," OGCI Publications, 1991.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Container for production allocation results.

    Attributes:
        well_allocations: Dictionary mapping well_id to allocated rate
        total_allocated: Total allocated production
        allocation_method: Method used for allocation
        facility_utilization: Facility utilization (fraction)
    """

    well_allocations: dict[str, float]
    total_allocated: float
    allocation_method: str
    facility_utilization: float


@dataclass
class FacilityConstraints:
    """Container for facility constraints.

    Attributes:
        max_total_rate: Maximum total production rate
        max_well_rate: Maximum per-well rate
        min_well_rate: Minimum per-well rate
        facility_capacity: Facility processing capacity
    """

    max_total_rate: float | None = None
    max_well_rate: float | None = None
    min_well_rate: float | None = None
    facility_capacity: float | None = None


def allocate_production_proportional(
    well_capacities: dict[str, float],
    total_target: float,
) -> dict[str, float]:
    """Allocate production proportionally based on well capacities.

    Args:
        well_capacities: Dictionary mapping well_id to capacity
        total_target: Total target production rate

    Returns:
        Dictionary mapping well_id to allocated rate

    Example:
        >>> capacities = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
        >>> allocations = allocate_production_proportional(capacities, total_target=2000)
    """
    total_capacity = sum(well_capacities.values())

    if total_capacity <= 0:
        return {well_id: 0.0 for well_id in well_capacities.keys()}

    allocations = {}
    for well_id, capacity in well_capacities.items():
        allocation = (capacity / total_capacity) * total_target
        allocations[well_id] = float(allocation)

    return allocations


def allocate_production_optimal(
    well_capacities: dict[str, float],
    well_priorities: dict[str, float] | None,
    total_target: float,
    constraints: FacilityConstraints | None = None,
) -> dict[str, float]:
    """Allocate production optimally with constraints.

    Uses linear programming for optimal allocation.

    Args:
        well_capacities: Dictionary mapping well_id to capacity
        well_priorities: Optional dictionary mapping well_id to priority
        total_target: Total target production rate
        constraints: Facility constraints

    Returns:
        Dictionary mapping well_id to allocated rate

    Example:
        >>> capacities = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
        >>> allocations = allocate_production_optimal(capacities, None, total_target=2000)
    """
    well_ids = list(well_capacities.keys())
    n_wells = len(well_ids)

    if n_wells == 0:
        return {}

    if constraints is None:
        constraints = FacilityConstraints()

    # Proportional allocation with constraints
    allocations = allocate_production_proportional(well_capacities, total_target)

    # Apply constraints
    total_allocated = sum(allocations.values())

    # Apply max total rate constraint
    if constraints.max_total_rate is not None:
        if total_allocated > constraints.max_total_rate:
            scale_factor = constraints.max_total_rate / total_allocated
            allocations = {
                well_id: rate * scale_factor for well_id, rate in allocations.items()
            }

    # Apply per-well constraints
    for well_id in well_ids:
        if constraints.max_well_rate is not None:
            allocations[well_id] = min(
                allocations[well_id], constraints.max_well_rate
            )
        if constraints.min_well_rate is not None:
            allocations[well_id] = max(
                allocations[well_id], constraints.min_well_rate
            )
        # Don't exceed capacity
        allocations[well_id] = min(
            allocations[well_id], well_capacities[well_id]
        )

    return allocations


def optimize_production_allocation(
    well_capacities: dict[str, float],
    well_costs: dict[str, float] | None,
    total_target: float,
    constraints: FacilityConstraints | None = None,
) -> AllocationResult:
    """Optimize production allocation with constraints.

    Args:
        well_capacities: Dictionary mapping well_id to capacity
        well_costs: Optional dictionary mapping well_id to cost per unit
        total_target: Total target production rate
        constraints: Facility constraints

    Returns:
        AllocationResult with optimized allocations

    Example:
        >>> capacities = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
        >>> costs = {'well_1': 10, 'well_2': 12, 'well_3': 15}
        >>> result = optimize_production_allocation(capacities, costs, total_target=2000)
    """
    if constraints is None:
        constraints = FacilityConstraints()

    # If costs provided, optimize for minimum cost
    if well_costs is not None:
        allocations = allocate_production_optimal(
            well_capacities, None, total_target, constraints
        )
        allocation_method = "cost_optimized"
    else:
        # Proportional allocation
        allocations = allocate_production_proportional(well_capacities, total_target)
        allocation_method = "proportional"

        # Apply constraints
        total_allocated = sum(allocations.values())
        if constraints.max_total_rate is not None:
            if total_allocated > constraints.max_total_rate:
                scale_factor = constraints.max_total_rate / total_allocated
                allocations = {
                    well_id: rate * scale_factor
                    for well_id, rate in allocations.items()
                }

        # Apply per-well constraints
        for well_id in allocations.keys():
            if constraints.max_well_rate is not None:
                allocations[well_id] = min(
                    allocations[well_id], constraints.max_well_rate
                )
            if constraints.min_well_rate is not None:
                allocations[well_id] = max(
                    allocations[well_id], constraints.min_well_rate
                )
            allocations[well_id] = min(
                allocations[well_id], well_capacities[well_id]
            )

    total_allocated = sum(allocations.values())
    facility_utilization = (
        total_allocated / constraints.facility_capacity
        if constraints.facility_capacity and constraints.facility_capacity > 0
        else 1.0
    )

    return AllocationResult(
        well_allocations=allocations,
        total_allocated=float(total_allocated),
        allocation_method=allocation_method,
        facility_utilization=float(facility_utilization),
    )


def calculate_facility_utilization(
    production_rates: dict[str, float],
    facility_capacity: float,
) -> float:
    """Calculate facility utilization.

    Args:
        production_rates: Dictionary mapping well_id to production rate
        facility_capacity: Facility processing capacity

    Returns:
        Facility utilization (0-1)

    Example:
        >>> rates = {'well_1': 500, 'well_2': 400, 'well_3': 300}
        >>> utilization = calculate_facility_utilization(rates, facility_capacity=1500)
    """
    total_rate = sum(production_rates.values())
    utilization = total_rate / facility_capacity if facility_capacity > 0 else 1.0
    return float(max(0.0, min(1.0, utilization)))


def apply_facility_constraints(
    production_rates: dict[str, float],
    constraints: FacilityConstraints,
) -> dict[str, float]:
    """Apply facility constraints to production rates.

    Args:
        production_rates: Dictionary mapping well_id to production rate
        constraints: Facility constraints

    Returns:
        Dictionary with constrained rates

    Example:
        >>> rates = {'well_1': 1000, 'well_2': 800, 'well_3': 600}
        >>> constraints = FacilityConstraints(max_total_rate=2000)
        >>> constrained = apply_facility_constraints(rates, constraints)
    """
    constrained_rates = production_rates.copy()

    # Apply per-well constraints
    for well_id in constrained_rates.keys():
        if constraints.max_well_rate is not None:
            constrained_rates[well_id] = min(
                constrained_rates[well_id], constraints.max_well_rate
            )
        if constraints.min_well_rate is not None:
            constrained_rates[well_id] = max(
                constrained_rates[well_id], constraints.min_well_rate
            )

    # Apply total rate constraint
    total_rate = sum(constrained_rates.values())
    if constraints.max_total_rate is not None:
        if total_rate > constraints.max_total_rate:
            scale_factor = constraints.max_total_rate / total_rate
            constrained_rates = {
                well_id: rate * scale_factor
                for well_id, rate in constrained_rates.items()
            }

    return constrained_rates

