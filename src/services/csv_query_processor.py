#!/usr/bin/env python3
"""
CSV Query Processor - Process natural language queries against CSV data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CSVQueryProcessor:
    """Processes natural language queries against CSV data for RBA agents"""
    
    def __init__(self):
        self.query_patterns = {
            # Pipeline queries
            "stale_opportunities": [
                r"stale.*opportunities?", r"old.*deals?", r"inactive.*pipeline",
                r"opportunities?.*haven't.*touched", r"deals?.*no.*activity"
            ],
            "pipeline_health": [
                r"pipeline.*health", r"pipeline.*status", r"how.*pipeline.*doing",
                r"pipeline.*analysis", r"pipeline.*overview"
            ],
            "deal_amounts": [
                r"deal.*amounts?", r"opportunity.*values?", r"pipeline.*value",
                r"total.*pipeline", r"deal.*sizes?"
            ],
            
            # Forecast queries
            "forecast_accuracy": [
                r"forecast.*accuracy", r"forecast.*variance", r"forecast.*vs.*actual",
                r"how.*accurate.*forecast", r"forecast.*performance"
            ],
            "quota_attainment": [
                r"quota.*attainment", r"quota.*achievement", r"quota.*progress",
                r"target.*vs.*actual", r"sales.*targets?"
            ],
            
            # Account queries
            "account_health": [
                r"account.*health", r"customer.*health", r"account.*risk",
                r"churn.*risk", r"at.*risk.*accounts?"
            ],
            "account_analysis": [
                r"account.*analysis", r"customer.*analysis", r"account.*overview",
                r"account.*summary", r"customer.*insights?"
            ],
            
            # Revenue queries
            "revenue_recognition": [
                r"revenue.*recognition", r"recognize.*revenue", r"revenue.*processing",
                r"closed.*won.*revenue", r"booking.*revenue"
            ],
            "revenue_analysis": [
                r"revenue.*analysis", r"revenue.*breakdown", r"revenue.*summary",
                r"total.*revenue", r"revenue.*metrics?"
            ],
            
            # General queries
            "data_summary": [
                r"data.*summary", r"overview", r"summary", r"what.*data",
                r"show.*me.*data", r"analyze.*data"
            ]
        }
    
    async def process_natural_language_query(
        self, 
        csv_data: pd.DataFrame, 
        user_query: str, 
        data_type: str
    ) -> Dict[str, Any]:
        """
        Process natural language query and return enhanced query with context
        """
        try:
            logger.info(f"🗣️ Processing query: '{user_query}'")
            
            # Normalize query
            normalized_query = user_query.lower().strip()
            
            # Detect query intent
            query_intent = self._detect_query_intent(normalized_query)
            
            # Analyze CSV data for context
            data_context = await self._analyze_data_context(csv_data, data_type)
            
            # Generate enhanced query
            enhanced_query = self._generate_enhanced_query(
                original_query=user_query,
                query_intent=query_intent,
                data_context=data_context,
                csv_data=csv_data
            )
            
            # Execute specific analysis based on intent
            query_results = await self._execute_query_analysis(
                csv_data=csv_data,
                query_intent=query_intent,
                data_context=data_context
            )
            
            return {
                "original_query": user_query,
                "normalized_query": normalized_query,
                "query_intent": query_intent,
                "enhanced_query": enhanced_query,
                "data_context": data_context,
                "query_results": query_results,
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "original_query": user_query,
                "error": str(e),
                "processing_failed": True
            }
    
    def _detect_query_intent(self, normalized_query: str) -> str:
        """Detect the intent of the user's query"""
        
        # Check each pattern category
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query):
                    return intent
        
        # Default intent based on keywords
        if any(keyword in normalized_query for keyword in ["pipeline", "opportunity", "deal"]):
            return "pipeline_health"
        elif any(keyword in normalized_query for keyword in ["forecast", "prediction", "quota"]):
            return "forecast_accuracy"
        elif any(keyword in normalized_query for keyword in ["account", "customer", "client"]):
            return "account_analysis"
        elif any(keyword in normalized_query for keyword in ["revenue", "money", "amount"]):
            return "revenue_analysis"
        else:
            return "data_summary"
    
    async def _analyze_data_context(self, csv_data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Analyze CSV data to provide context for query processing"""
        
        context = {
            "total_rows": len(csv_data),
            "total_columns": len(csv_data.columns),
            "columns": list(csv_data.columns),
            "data_type": data_type
        }
        
        # Find key columns based on data type
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Amount/value columns
        amount_cols = [col for col in csv_data.columns if any(keyword in col.lower() for keyword in ['amount', 'value', 'revenue', 'price', 'cost'])]
        if amount_cols:
            context["amount_columns"] = amount_cols
        
        # Date columns
        date_cols = [col for col in csv_data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'close'])]
        if date_cols:
            context["date_columns"] = date_cols
        
        # Status/stage columns
        status_cols = [col for col in csv_data.columns if any(keyword in col.lower() for keyword in ['status', 'stage', 'state', 'phase'])]
        if status_cols:
            context["status_columns"] = status_cols
        
        # ID columns
        id_cols = [col for col in csv_data.columns if 'id' in col.lower()]
        if id_cols:
            context["id_columns"] = id_cols
        
        return context
    
    def _generate_enhanced_query(
        self, 
        original_query: str, 
        query_intent: str, 
        data_context: Dict[str, Any], 
        csv_data: pd.DataFrame
    ) -> str:
        """Generate an enhanced query with specific context"""
        
        base_query = original_query
        
        # Add data context
        data_info = f"analyzing {data_context['total_rows']} {data_context['data_type']} records"
        
        # Enhance based on intent
        if query_intent == "stale_opportunities":
            if "date_columns" in data_context:
                enhanced_query = f"Identify stale opportunities from {data_info} using date columns {data_context['date_columns']}"
            else:
                enhanced_query = f"Analyze pipeline hygiene for {data_info}"
        
        elif query_intent == "pipeline_health":
            enhanced_query = f"Analyze overall pipeline health from {data_info}"
            if "status_columns" in data_context:
                enhanced_query += f" focusing on stages in {data_context['status_columns']}"
        
        elif query_intent == "forecast_accuracy":
            enhanced_query = f"Validate forecast accuracy from {data_info}"
            if "amount_columns" in data_context:
                enhanced_query += f" using amounts in {data_context['amount_columns']}"
        
        elif query_intent == "account_health":
            enhanced_query = f"Assess account health from {data_info}"
        
        elif query_intent == "revenue_recognition":
            enhanced_query = f"Process revenue recognition for {data_info}"
            if "amount_columns" in data_context:
                enhanced_query += f" using revenue data in {data_context['amount_columns']}"
        
        elif query_intent == "data_summary":
            enhanced_query = f"Provide comprehensive analysis of {data_info}"
        
        else:
            enhanced_query = f"{base_query} for {data_info}"
        
        return enhanced_query
    
    async def _execute_query_analysis(
        self, 
        csv_data: pd.DataFrame, 
        query_intent: str, 
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific analysis based on query intent"""
        
        try:
            if query_intent == "stale_opportunities":
                return await self._analyze_stale_opportunities(csv_data, data_context)
            
            elif query_intent == "pipeline_health":
                return await self._analyze_pipeline_health(csv_data, data_context)
            
            elif query_intent == "deal_amounts":
                return await self._analyze_deal_amounts(csv_data, data_context)
            
            elif query_intent == "forecast_accuracy":
                return await self._analyze_forecast_accuracy(csv_data, data_context)
            
            elif query_intent == "account_health":
                return await self._analyze_account_health(csv_data, data_context)
            
            elif query_intent == "revenue_analysis":
                return await self._analyze_revenue(csv_data, data_context)
            
            else:
                return await self._analyze_general_data(csv_data, data_context)
                
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            return {"error": str(e), "analysis_failed": True}
    
    async def _analyze_stale_opportunities(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stale opportunities"""
        
        results = {"analysis_type": "stale_opportunities"}
        
        if "date_columns" in data_context:
            current_date = datetime.now()
            stale_count = 0
            very_stale_count = 0
            
            for date_col in data_context["date_columns"]:
                try:
                    dates = pd.to_datetime(csv_data[date_col], errors='coerce')
                    valid_dates = dates.dropna()
                    
                    if len(valid_dates) > 0:
                        days_since = (current_date - valid_dates).dt.days
                        stale_count += (days_since > 30).sum()  # 30+ days
                        very_stale_count += (days_since > 90).sum()  # 90+ days
                        
                except Exception as e:
                    logger.warning(f"Could not process date column {date_col}: {e}")
            
            results.update({
                "stale_opportunities": int(stale_count),
                "very_stale_opportunities": int(very_stale_count),
                "total_opportunities": len(csv_data),
                "staleness_percentage": (stale_count / len(csv_data)) * 100 if len(csv_data) > 0 else 0
            })
        
        return results
    
    async def _analyze_pipeline_health(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall pipeline health"""
        
        results = {"analysis_type": "pipeline_health"}
        
        # Stage analysis
        if "status_columns" in data_context:
            for status_col in data_context["status_columns"]:
                stage_counts = csv_data[status_col].value_counts().to_dict()
                results[f"{status_col}_distribution"] = stage_counts
        
        # Amount analysis
        if "amount_columns" in data_context:
            for amount_col in data_context["amount_columns"]:
                try:
                    amount_data = pd.to_numeric(csv_data[amount_col], errors='coerce')
                    if not amount_data.isna().all():
                        results[f"{amount_col}_metrics"] = {
                            "total": float(amount_data.sum()),
                            "average": float(amount_data.mean()),
                            "count": int(amount_data.count())
                        }
                except Exception as e:
                    logger.warning(f"Could not process amount column {amount_col}: {e}")
        
        results["total_records"] = len(csv_data)
        return results
    
    async def _analyze_deal_amounts(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deal amounts and values"""
        
        results = {"analysis_type": "deal_amounts"}
        
        if "amount_columns" in data_context:
            total_value = 0
            deal_count = 0
            
            for amount_col in data_context["amount_columns"]:
                try:
                    amount_data = pd.to_numeric(csv_data[amount_col], errors='coerce')
                    valid_amounts = amount_data.dropna()
                    
                    if len(valid_amounts) > 0:
                        total_value += valid_amounts.sum()
                        deal_count += len(valid_amounts)
                        
                        results[f"{amount_col}_analysis"] = {
                            "total": float(valid_amounts.sum()),
                            "average": float(valid_amounts.mean()),
                            "median": float(valid_amounts.median()),
                            "max": float(valid_amounts.max()),
                            "min": float(valid_amounts.min()),
                            "count": len(valid_amounts)
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not process amount column {amount_col}: {e}")
            
            results.update({
                "total_pipeline_value": float(total_value),
                "total_deals": int(deal_count),
                "average_deal_size": float(total_value / deal_count) if deal_count > 0 else 0
            })
        
        return results
    
    async def _analyze_forecast_accuracy(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze forecast accuracy"""
        
        results = {"analysis_type": "forecast_accuracy"}
        
        # Look for forecast vs actual columns
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        forecast_col = None
        actual_col = None
        
        for col_lower, col_original in columns_lower.items():
            if any(keyword in col_lower for keyword in ['forecast', 'predicted', 'target']):
                forecast_col = col_original
            elif any(keyword in col_lower for keyword in ['actual', 'achieved', 'closed']):
                actual_col = col_original
        
        if forecast_col and actual_col:
            try:
                forecast_data = pd.to_numeric(csv_data[forecast_col], errors='coerce')
                actual_data = pd.to_numeric(csv_data[actual_col], errors='coerce')
                
                # Calculate variance
                combined = pd.DataFrame({'forecast': forecast_data, 'actual': actual_data}).dropna()
                
                if len(combined) > 0:
                    variance = ((combined['actual'] - combined['forecast']) / combined['forecast'] * 100).fillna(0)
                    
                    results.update({
                        "forecast_vs_actual_analysis": {
                            "total_forecasts": len(combined),
                            "average_variance_percentage": float(variance.mean()),
                            "accuracy_within_10_percent": int((abs(variance) <= 10).sum()),
                            "over_forecast_count": int((variance > 0).sum()),
                            "under_forecast_count": int((variance < 0).sum()),
                            "total_forecast_amount": float(combined['forecast'].sum()),
                            "total_actual_amount": float(combined['actual'].sum())
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Could not analyze forecast accuracy: {e}")
        
        return results
    
    async def _analyze_account_health(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze account health"""
        
        results = {"analysis_type": "account_health"}
        
        # Look for health score columns
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        health_col = None
        for col_lower, col_original in columns_lower.items():
            if any(keyword in col_lower for keyword in ['health', 'score', 'risk']):
                health_col = col_original
                break
        
        if health_col:
            try:
                health_data = pd.to_numeric(csv_data[health_col], errors='coerce')
                valid_health = health_data.dropna()
                
                if len(valid_health) > 0:
                    # Determine scale (0-1 or 0-100)
                    max_val = valid_health.max()
                    if max_val <= 1:
                        # 0-1 scale
                        at_risk_threshold = 0.5
                        healthy_threshold = 0.8
                    else:
                        # 0-100 scale
                        at_risk_threshold = 50
                        healthy_threshold = 80
                    
                    results.update({
                        "health_score_analysis": {
                            "total_accounts": len(valid_health),
                            "average_health_score": float(valid_health.mean()),
                            "at_risk_accounts": int((valid_health < at_risk_threshold).sum()),
                            "healthy_accounts": int((valid_health >= healthy_threshold).sum()),
                            "health_distribution": {
                                "min": float(valid_health.min()),
                                "max": float(valid_health.max()),
                                "median": float(valid_health.median())
                            }
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Could not analyze health scores: {e}")
        
        return results
    
    async def _analyze_revenue(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze revenue data"""
        
        results = {"analysis_type": "revenue_analysis"}
        
        if "amount_columns" in data_context:
            total_revenue = 0
            
            for amount_col in data_context["amount_columns"]:
                try:
                    revenue_data = pd.to_numeric(csv_data[amount_col], errors='coerce')
                    valid_revenue = revenue_data.dropna()
                    
                    if len(valid_revenue) > 0:
                        col_total = valid_revenue.sum()
                        total_revenue += col_total
                        
                        results[f"{amount_col}_revenue"] = {
                            "total": float(col_total),
                            "average": float(valid_revenue.mean()),
                            "count": len(valid_revenue)
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not process revenue column {amount_col}: {e}")
            
            results["total_revenue"] = float(total_revenue)
        
        return results
    
    async def _analyze_general_data(self, csv_data: pd.DataFrame, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general data analysis"""
        
        results = {
            "analysis_type": "general_data_summary",
            "total_records": len(csv_data),
            "total_columns": len(csv_data.columns),
            "column_types": csv_data.dtypes.to_dict(),
            "missing_data": csv_data.isnull().sum().to_dict(),
            "sample_data": csv_data.head(3).fillna('').to_dict('records')
        }
        
        return results
