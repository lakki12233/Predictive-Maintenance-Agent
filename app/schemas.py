# app/schemas.py
"""Pydantic schemas for API request/response validation."""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class SensorSample(BaseModel):
    """Single sensor reading at a point in time."""
    ts: str | None = None
    # Flexible sensor fields (sensor_00 through sensor_51)
    
    model_config = ConfigDict(extra="allow")  # Allow dynamic sensor_XX fields


class EnvironmentalData(BaseModel):
    """Environmental and transactional context for risk assessment."""
    operating_hours: float | None = Field(None, ge=0, description="Total hours since last overhaul")
    days_since_last_maintenance: float | None = Field(None, ge=0, description="Days since routine maintenance")
    ambient_temperature_c: float | None = Field(None, ge=-40, le=80, description="Ambient temperature in Celsius")
    ambient_humidity_percent: float | None = Field(None, ge=0, le=100, description="Relative humidity percentage")
    load_factor: float | None = Field(None, ge=0, le=2.0, description="Current load vs rated capacity (0-1 normal)")
    maintenance_overdue: bool | None = Field(None, description="Whether scheduled maintenance is overdue")


class PredictRequest(BaseModel):
    """Input schema for single prediction request."""
    asset_id: str = Field(..., min_length=1, description="Unique asset identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the measurement")
    sensor_window: list[SensorSample] = Field(..., min_length=1, description="Time-series sensor readings")
    image_base64: str | None = Field(None, description="Base64-encoded JPEG/PNG image for rust detection")
    environmental: EnvironmentalData | None = Field(None, description="Operational context for risk adjustment")


class PredictResponse(BaseModel):
    """Output schema for prediction response."""
    # Fix Pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())
    
    asset_id: str
    failure_probability: float = Field(..., ge=0, le=1, description="Likelihood of failure (0-1)")
    estimated_time_to_breakdown_hours: float = Field(..., ge=0, description="Predicted time to failure")
    predicted_fault_type: Optional[str] = Field(None, description="Most likely fault category")
    fault_confidence: float = Field(..., ge=0, le=1, description="Confidence in fault prediction")
    top_signals: List[str] = Field(..., max_length=10, description="Top contributing factors")
    inference_ms: int = Field(..., ge=0, description="Total inference time in milliseconds")
    model_version: str = Field(..., description="Version identifier for models used")
    explanation: Optional[str] = Field(None, description="Human-readable explanation of prediction")
