"""Federated-learning core for the 3D monitoring platform."""

from fl_core.events import FLEvent
from fl_core.strategy import MonitoringFedAvg

__all__ = ["FLEvent", "MonitoringFedAvg"]
