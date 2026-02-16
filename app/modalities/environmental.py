"""
Environmental and Transactional Modality
=========================================
Analyzes operational context: maintenance history, operating conditions,
load factors, etc. to adjust failure predictions.
"""
from typing import Tuple, List


class EnvironmentalInference:
    """
    Rule-based environmental risk scorer.
    Returns risk adjustment factor and contributing signals.
    """
    
    def __init__(self):
        self.version = "env_v1"
        
        # Thresholds (configurable)
        self.HOURS_HIGH_RISK = 2000  # operating hours
        self.HOURS_MEDIUM_RISK = 1000
        self.DAYS_OVERDUE_HIGH = 60
        self.DAYS_OVERDUE_MEDIUM = 30
        self.TEMP_HIGH_C = 45
        self.TEMP_LOW_C = 5
        self.HUMIDITY_HIGH = 85
        self.LOAD_HIGH = 0.95
        
    def predict(self, env_data) -> Tuple[float, List[str]]:
        """
        Returns:
            risk_multiplier: (0.5 to 2.0) - multiplier for failure probability
            signals: list of contributing environmental factors
        """
        if env_data is None:
            return 1.0, []
        
        risk_multiplier = 1.0
        signals = []
        
        # Operating hours risk
        if env_data.operating_hours is not None:
            hours = float(env_data.operating_hours)
            if hours >= self.HOURS_HIGH_RISK:
                risk_multiplier *= 1.4
                signals.append(f"env:high_hours({hours:.0f}h)")
            elif hours >= self.HOURS_MEDIUM_RISK:
                risk_multiplier *= 1.2
                signals.append(f"env:medium_hours({hours:.0f}h)")
            else:
                signals.append(f"env:normal_hours({hours:.0f}h)")
                
        # Maintenance overdue
        if env_data.days_since_last_maintenance is not None:
            days = float(env_data.days_since_last_maintenance)
            overdue = bool(env_data.maintenance_overdue) if env_data.maintenance_overdue is not None else (days > 90)
            
            if overdue and days >= self.DAYS_OVERDUE_HIGH:
                risk_multiplier *= 1.5
                signals.append(f"env:maint_critical({days:.0f}d)")
            elif overdue or days >= self.DAYS_OVERDUE_MEDIUM:
                risk_multiplier *= 1.3
                signals.append(f"env:maint_overdue({days:.0f}d)")
            else:
                signals.append(f"env:maint_ok({days:.0f}d)")
                
        # Ambient temperature
        if env_data.ambient_temperature_c is not None:
            temp = float(env_data.ambient_temperature_c)
            if temp >= self.TEMP_HIGH_C:
                risk_multiplier *= 1.15
                signals.append(f"env:temp_high({temp:.1f}C)")
            elif temp <= self.TEMP_LOW_C:
                risk_multiplier *= 1.1
                signals.append(f"env:temp_low({temp:.1f}C)")
            else:
                signals.append(f"env:temp_ok({temp:.1f}C)")
                
        # Humidity
        if env_data.ambient_humidity_percent is not None:
            humidity = float(env_data.ambient_humidity_percent)
            if humidity >= self.HUMIDITY_HIGH:
                risk_multiplier *= 1.15
                signals.append(f"env:humidity_high({humidity:.0f}%)")
            else:
                signals.append(f"env:humidity_ok({humidity:.0f}%)")
                
        # Load factor
        if env_data.load_factor is not None:
            load = float(env_data.load_factor)
            if load >= self.LOAD_HIGH:
                risk_multiplier *= 1.2
                signals.append(f"env:overloaded({load:.2f})")
            elif load >= 0.8:
                risk_multiplier *= 1.1
                signals.append(f"env:high_load({load:.2f})")
            else:
                signals.append(f"env:normal_load({load:.2f})")
        
        # Clamp multiplier to reasonable range
        risk_multiplier = max(0.5, min(2.0, risk_multiplier))
        
        # Add summary signal
        if risk_multiplier >= 1.5:
            signals.insert(0, "env:conditions_critical")
        elif risk_multiplier >= 1.2:
            signals.insert(0, "env:conditions_elevated")
        elif risk_multiplier >= 1.0:
            signals.insert(0, "env:conditions_normal")
        else:
            signals.insert(0, "env:conditions_favorable")
            
        return risk_multiplier, signals
