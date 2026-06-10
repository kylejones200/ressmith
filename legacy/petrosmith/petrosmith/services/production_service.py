"""
Production service - orchestrates production operations.

This service layer coordinates production models and core calculations.
"""

import math
import logging
from typing import List, Dict, Optional, Tuple, Any
from petrosmith.models import Well, FluidProperties
from petrosmith.core import ProductionCalculations, ReservoirCalculations

logger = logging.getLogger(__name__)


class ProductionService:
    """
    Service for managing production operations and optimization.
    
    Orchestrates production data, calculations, and workflows.
    """
    
    def __init__(self):
        """Initialize production service with data storage."""
        self.production_data: Dict[str, List[Dict]] = {}
        self.well_tests: Dict[str, Dict] = {}
    
    def add_production_data(
        self,
        well_id: str,
        date: str,
        oil_rate: float,
        gas_rate: float,
        water_rate: float
    ) -> None:
        """
        Add daily production data for a well.
        
        Args:
            well_id: Well identifier
            date: Production date
            oil_rate: Oil rate in STB/day
            gas_rate: Gas rate in Mscf/day
            water_rate: Water rate in STB/day
        """
        if well_id not in self.production_data:
            self.production_data[well_id] = []
        
        self.production_data[well_id].append({
            "date": date,
            "oil_rate": oil_rate,
            "gas_rate": gas_rate,
            "water_rate": water_rate
        })
    
    def analyze_well_performance(
        self,
        well_id: str
    ) -> Dict[str, Any]:
        """
        Analyze well production performance.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dictionary with performance analysis
        """
        if well_id not in self.production_data or not self.production_data[well_id]:
            raise ValueError(f"No production data for well {well_id}")
        
        data = self.production_data[well_id]
        
        # Get latest production
        latest = data[-1]
        
        # Calculate water cut
        water_cut = ProductionCalculations.calculate_water_cut(
            water_production=latest["water_rate"],
            oil_production=latest["oil_rate"]
        )
        
        # Calculate GOR
        gor = ProductionCalculations.calculate_gor(
            gas_production=latest["gas_rate"],
            oil_production=latest["oil_rate"]
        )
        
        # Calculate averages over last 30 days (or available data)
        recent_data = data[-30:] if len(data) >= 30 else data
        avg_oil = sum(d["oil_rate"] for d in recent_data) / len(recent_data)
        avg_gas = sum(d["gas_rate"] for d in recent_data) / len(recent_data)
        avg_water = sum(d["water_rate"] for d in recent_data) / len(recent_data)
        
        # Estimate decline if enough data
        decline_rate = 0.0
        if len(data) >= 365:
            first_year = data[:365]
            last_year = data[-365:]
            initial_avg = sum(d["oil_rate"] for d in first_year) / len(first_year)
            recent_avg = sum(d["oil_rate"] for d in last_year) / len(last_year)
            
            if initial_avg > 0:
                decline_rate = (initial_avg - recent_avg) / initial_avg
        
        return {
            "well_id": well_id,
            "current_production": {
                "oil_rate": latest["oil_rate"],
                "gas_rate": latest["gas_rate"],
                "water_rate": latest["water_rate"],
                "total_liquid": latest["oil_rate"] + latest["water_rate"]
            },
            "performance_indicators": {
                "water_cut": water_cut,
                "gor": gor,
                "decline_rate_annual": decline_rate
            },
            "averages_30_day": {
                "oil": avg_oil,
                "gas": avg_gas,
                "water": avg_water
            },
            "status": self._assess_well_status(water_cut, latest["oil_rate"], decline_rate)
        }
    
    def _assess_well_status(
        self,
        water_cut: float,
        oil_rate: float,
        decline_rate: float
    ) -> str:
        """Assess overall well status."""
        if water_cut > 95:
            return "poor_high_water"
        elif oil_rate < 10:
            return "poor_low_rate"
        elif decline_rate > 0.3:
            return "declining_fast"
        elif decline_rate > 0.15:
            return "declining_moderate"
        elif water_cut > 80:
            return "fair_high_water"
        else:
            return "good"
    
    def forecast_production(
        self,
        well_id: str,
        forecast_years: int = 5,
        use_dca: bool = True,
        model: str = "arps",
        kind: str = "hyperbolic",
    ) -> Dict[str, Any]:
        """
        Forecast future production using decline curve analysis.

        When the optional ``decline-curve`` library is installed
        (``pip install petrosmith[dca]``), uses it for Arps (exponential,
        hyperbolic, harmonic) or other models. Otherwise falls back to
        built-in exponential decline.

        Args:
            well_id: Well identifier
            forecast_years: Number of years to forecast
            use_dca: If True (default), use decline-curve library when available
            model: When using DCA: 'arps', 'arima', 'timesfm', 'chronos'
            kind: When using Arps: 'exponential', 'harmonic', or 'hyperbolic'

        Returns:
            Dictionary with production forecast (forecast, current_rate,
            decline_rate_annual, economic_limit; plus dca_params, model, kind
            when DCA was used)
        """
        if well_id not in self.production_data or not self.production_data[well_id]:
            raise ValueError(f"No production data for well {well_id}")

        data = self.production_data[well_id]

        if use_dca:
            try:
                from petrosmith.integrations.decline_curve import forecast_with_dca

                result = forecast_with_dca(
                    production_data=data,
                    forecast_years=forecast_years,
                    model=model,
                    kind=kind,
                )
                result["well_id"] = well_id
                return result
            except ImportError:
                logger.debug(
                    "decline-curve not installed; falling back to exponential. "
                    "Install with: pip install petrosmith[dca]"
                )
            except ValueError as e:
                # Not enough data or fit failed
                logger.debug("DCA failed (%s), falling back to exponential", e)

        # Fallback: built-in exponential decline
        if len(data) < 90:
            raise ValueError("Need at least 90 days of data for reliable forecast")

        first_30 = data[:30]
        last_30 = data[-30:]
        initial_rate = sum(d["oil_rate"] for d in first_30) / len(first_30)
        current_rate = sum(d["oil_rate"] for d in last_30) / len(last_30)
        days_between = len(data)
        years_between = days_between / 365

        if years_between > 0 and initial_rate > 0:
            decline_rate = -math.log(current_rate / initial_rate) / years_between
        else:
            decline_rate = 0.10

        forecast = []
        cumulative = 0.0
        for year in range(forecast_years + 1):
            rate = ProductionCalculations.calculate_decline_curve_exponential(
                initial_rate=current_rate,
                decline_rate=decline_rate,
                time=year,
            )
            if year > 0:
                year_cumulative = (
                    ProductionCalculations.calculate_cumulative_production_exponential(
                        initial_rate=current_rate,
                        decline_rate=decline_rate,
                        time=year,
                    )
                )
                cumulative = year_cumulative
            forecast.append({"year": year, "rate": rate, "cumulative": cumulative})

        return {
            "well_id": well_id,
            "current_rate": current_rate,
            "decline_rate_annual": decline_rate,
            "forecast": forecast,
            "economic_limit": current_rate * 0.1,
        }
    
    def optimize_artificial_lift(
        self,
        well_id: str,
        reservoir_pressure: float,
        depth: float,
        desired_rate: float,
        fluid_properties: FluidProperties
    ) -> Dict[str, Any]:
        """
        Optimize artificial lift system for a well.
        
        Args:
            well_id: Well identifier
            reservoir_pressure: Reservoir pressure in psi
            depth: Well depth in feet
            desired_rate: Desired production rate in STB/day
            fluid_properties: Fluid properties
            
        Returns:
            Dictionary with lift optimization
        """
        # Calculate ESP requirements
        esp_head = ProductionCalculations.calculate_esp_required_head(
            depth=depth,
            wellhead_pressure=100.0,  # Target 100 psi wellhead
            flow_rate=desired_rate,
            fluid_specific_gravity=fluid_properties.density / 8.34  # Convert ppg to SG
        )
        
        esp_hp = ProductionCalculations.calculate_esp_horsepower(
            flow_rate=desired_rate,
            total_head=esp_head,
            efficiency=0.70
        )
        
        # Gas lift analysis
        gas_lift_efficiency = ProductionCalculations.calculate_gas_lift_performance(
            injection_rate=desired_rate * 0.5,  # 500 scf per barrel
            injection_pressure=reservoir_pressure * 0.8,
            operating_pressure=reservoir_pressure * 0.5,
            liquid_rate=desired_rate
        )
        
        # Determine best option
        recommendations = []
        
        if depth < 8000 and desired_rate < 500:
            recommendations.append("Rod pump - suitable for shallow, low-rate wells")
            best_option = "rod_pump"
        elif esp_hp < 150 and depth > 5000:
            recommendations.append("ESP - good efficiency for this depth and rate")
            best_option = "esp"
        elif depth < 12000:
            recommendations.append("Gas lift - flexible and reliable option")
            best_option = "gas_lift"
        else:
            recommendations.append("ESP with high-stage pump recommended")
            best_option = "esp_high_stage"
        
        return {
            "well_id": well_id,
            "recommended_system": best_option,
            "esp_analysis": {
                "required_head": esp_head,
                "required_hp": esp_hp,
                "operating_cost_per_day": esp_hp * 24 * 0.10
            },
            "gas_lift_analysis": {
                "efficiency": gas_lift_efficiency,
                "injection_rate": desired_rate * 0.5,
                "operating_cost_per_day": desired_rate * 0.5 * 3.0
            },
            "recommendations": recommendations
        }
