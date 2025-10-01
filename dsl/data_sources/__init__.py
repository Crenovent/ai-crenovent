"""
Data Sources Module
Handles integration with CRM, historical data, external data, and calculated metrics
"""

from .data_source_manager import data_source_manager, DataSourceManager, DataAvailability

__all__ = ['data_source_manager', 'DataSourceManager', 'DataAvailability']
