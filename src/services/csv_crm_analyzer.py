#!/usr/bin/env python3
"""
CSV CRM Analyzer - Analyze CRM data from CSV uploads for RBA workflows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class CSVCRMAnalyzer:
    """Analyzes CRM data from CSV uploads and prepares insights for RBA agents"""
    
    def __init__(self):
        self.supported_data_types = [
            "opportunities", "accounts", "contacts", "activities", 
            "forecasts", "revenue", "general_crm"
        ]
    
    async def analyze_crm_data(
        self, 
        csv_data: pd.DataFrame, 
        data_type: str, 
        analysis_type: str = "auto_detect",
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method that routes to specific analyzers based on data type
        """
        try:
            logger.info(f"🔍 Analyzing {data_type} data with {len(csv_data)} records")
            
            # Basic data quality analysis
            basic_analysis = self._analyze_data_quality(csv_data)
            
            # Type-specific analysis
            if data_type == "opportunities":
                specific_analysis = await self._analyze_opportunities(csv_data)
            elif data_type == "accounts":
                specific_analysis = await self._analyze_accounts(csv_data)
            elif data_type == "contacts":
                specific_analysis = await self._analyze_contacts(csv_data)
            elif data_type == "activities":
                specific_analysis = await self._analyze_activities(csv_data)
            elif data_type == "forecasts":
                specific_analysis = await self._analyze_forecasts(csv_data)
            elif data_type == "revenue":
                specific_analysis = await self._analyze_revenue(csv_data)
            else:
                specific_analysis = await self._analyze_general_crm(csv_data)
            
            # Combine analyses
            combined_analysis = {
                **basic_analysis,
                **specific_analysis,
                "data_type": data_type,
                "analysis_type": analysis_type,
                "user_query": user_query,
                "analyzed_at": datetime.now().isoformat()
            }
            
            # Add recommendations
            combined_analysis["recommendations"] = self._generate_recommendations(combined_analysis)
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"❌ CRM analysis failed: {e}")
            return {
                "error": str(e),
                "data_type": data_type,
                "analysis_failed": True
            }
    
    def _analyze_data_quality(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic data quality metrics"""
        
        return {
            "total_records": len(csv_data),
            "total_columns": len(csv_data.columns),
            "columns": list(csv_data.columns),
            "missing_data_percentage": (csv_data.isnull().sum().sum() / (len(csv_data) * len(csv_data.columns))) * 100,
            "duplicate_rows": csv_data.duplicated().sum(),
            "data_types": csv_data.dtypes.to_dict(),
            "memory_usage_mb": csv_data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    async def _analyze_opportunities(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze opportunity/pipeline data"""
        
        analysis = {}
        
        # Find relevant columns (case-insensitive)
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Stage analysis
        stage_col = self._find_column(columns_lower, ['stage', 'opportunity_stage', 'sales_stage', 'status'])
        if stage_col:
            stage_distribution = csv_data[stage_col].value_counts().to_dict()
            analysis["stage_distribution"] = stage_distribution
            analysis["total_opportunities"] = len(csv_data)
        
        # Amount analysis
        amount_col = self._find_column(columns_lower, ['amount', 'value', 'deal_size', 'opportunity_amount'])
        if amount_col:
            # Clean amount data (remove currency symbols, commas)
            amount_data = csv_data[amount_col].astype(str).str.replace(r'[$,]', '', regex=True)
            amount_data = pd.to_numeric(amount_data, errors='coerce')
            
            analysis["pipeline_value"] = {
                "total": float(amount_data.sum()) if not amount_data.isna().all() else 0,
                "average": float(amount_data.mean()) if not amount_data.isna().all() else 0,
                "median": float(amount_data.median()) if not amount_data.isna().all() else 0,
                "max": float(amount_data.max()) if not amount_data.isna().all() else 0,
                "min": float(amount_data.min()) if not amount_data.isna().all() else 0
            }
        
        # Date analysis for staleness
        date_cols = self._find_date_columns(csv_data)
        if date_cols:
            analysis["date_analysis"] = await self._analyze_opportunity_dates(csv_data, date_cols)
        
        # Win rate analysis
        if stage_col:
            closed_won_keywords = ['won', 'closed won', 'closed-won', 'success']
            closed_lost_keywords = ['lost', 'closed lost', 'closed-lost', 'failed']
            
            stages = csv_data[stage_col].str.lower().fillna('')
            won_count = sum(any(keyword in stage for keyword in closed_won_keywords) for stage in stages)
            lost_count = sum(any(keyword in stage for keyword in closed_lost_keywords) for stage in stages)
            
            if won_count + lost_count > 0:
                analysis["win_rate"] = won_count / (won_count + lost_count)
            
            analysis["closed_opportunities"] = {
                "won": won_count,
                "lost": lost_count,
                "total_closed": won_count + lost_count
            }
        
        return analysis
    
    async def _analyze_accounts(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze account data"""
        
        analysis = {}
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Account size analysis
        size_col = self._find_column(columns_lower, ['size', 'employees', 'company_size', 'employee_count'])
        if size_col:
            size_data = pd.to_numeric(csv_data[size_col], errors='coerce')
            analysis["account_sizes"] = {
                "average": float(size_data.mean()) if not size_data.isna().all() else 0,
                "median": float(size_data.median()) if not size_data.isna().all() else 0,
                "distribution": size_data.describe().to_dict() if not size_data.isna().all() else {}
            }
        
        # Industry analysis
        industry_col = self._find_column(columns_lower, ['industry', 'sector', 'vertical'])
        if industry_col:
            analysis["industry_distribution"] = csv_data[industry_col].value_counts().head(10).to_dict()
        
        # Health score analysis
        health_col = self._find_column(columns_lower, ['health', 'score', 'health_score', 'account_health'])
        if health_col:
            health_data = pd.to_numeric(csv_data[health_col], errors='coerce')
            if not health_data.isna().all():
                analysis["health_analysis"] = {
                    "average_health": float(health_data.mean()),
                    "at_risk_accounts": int((health_data < 50).sum()) if health_data.max() > 1 else int((health_data < 0.5).sum()),
                    "healthy_accounts": int((health_data >= 75).sum()) if health_data.max() > 1 else int((health_data >= 0.75).sum())
                }
        
        return analysis
    
    async def _analyze_contacts(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze contact data"""
        
        analysis = {}
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Email analysis
        email_col = self._find_column(columns_lower, ['email', 'email_address'])
        if email_col:
            emails = csv_data[email_col].dropna()
            valid_emails = emails[emails.str.contains(r'^[^@]+@[^@]+\.[^@]+$', regex=True, na=False)]
            analysis["email_analysis"] = {
                "total_contacts": len(csv_data),
                "contacts_with_email": len(emails),
                "valid_emails": len(valid_emails),
                "email_domains": valid_emails.str.split('@').str[1].value_counts().head(10).to_dict()
            }
        
        # Title analysis
        title_col = self._find_column(columns_lower, ['title', 'job_title', 'position'])
        if title_col:
            analysis["title_distribution"] = csv_data[title_col].value_counts().head(10).to_dict()
        
        return analysis
    
    async def _analyze_activities(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze activity data"""
        
        analysis = {}
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Activity type analysis
        type_col = self._find_column(columns_lower, ['type', 'activity_type', 'action'])
        if type_col:
            analysis["activity_types"] = csv_data[type_col].value_counts().to_dict()
        
        # Date analysis
        date_cols = self._find_date_columns(csv_data)
        if date_cols:
            analysis["activity_timeline"] = await self._analyze_activity_timeline(csv_data, date_cols)
        
        return analysis
    
    async def _analyze_forecasts(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze forecast data"""
        
        analysis = {}
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Forecast amount analysis
        forecast_col = self._find_column(columns_lower, ['forecast', 'predicted', 'target', 'quota'])
        actual_col = self._find_column(columns_lower, ['actual', 'achieved', 'closed'])
        
        if forecast_col and actual_col:
            forecast_data = pd.to_numeric(csv_data[forecast_col], errors='coerce')
            actual_data = pd.to_numeric(csv_data[actual_col], errors='coerce')
            
            if not forecast_data.isna().all() and not actual_data.isna().all():
                variance = ((actual_data - forecast_data) / forecast_data * 100).fillna(0)
                analysis["forecast_accuracy"] = {
                    "average_variance_percentage": float(variance.mean()),
                    "accuracy_rate": float((abs(variance) <= 10).mean() * 100),  # Within 10%
                    "over_forecast": int((variance > 0).sum()),
                    "under_forecast": int((variance < 0).sum())
                }
        
        return analysis
    
    async def _analyze_revenue(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze revenue data"""
        
        analysis = {}
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Revenue amount analysis
        revenue_col = self._find_column(columns_lower, ['revenue', 'amount', 'value', 'mrr', 'arr'])
        if revenue_col:
            revenue_data = pd.to_numeric(csv_data[revenue_col], errors='coerce')
            if not revenue_data.isna().all():
                analysis["revenue_metrics"] = {
                    "total_revenue": float(revenue_data.sum()),
                    "average_revenue": float(revenue_data.mean()),
                    "median_revenue": float(revenue_data.median()),
                    "revenue_distribution": revenue_data.describe().to_dict()
                }
        
        # Date-based revenue analysis
        date_cols = self._find_date_columns(csv_data)
        if date_cols and revenue_col:
            analysis["revenue_timeline"] = await self._analyze_revenue_timeline(csv_data, date_cols, revenue_col)
        
        return analysis
    
    async def _analyze_general_crm(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze general CRM data"""
        
        analysis = {}
        
        # Identify potential key fields
        columns_lower = {col.lower(): col for col in csv_data.columns}
        
        # Look for ID fields
        id_cols = [col for col in csv_data.columns if 'id' in col.lower()]
        if id_cols:
            analysis["id_fields"] = id_cols
        
        # Look for name fields
        name_cols = [col for col in csv_data.columns if any(keyword in col.lower() for keyword in ['name', 'title', 'subject'])]
        if name_cols:
            analysis["name_fields"] = name_cols
        
        # Look for date fields
        date_cols = self._find_date_columns(csv_data)
        if date_cols:
            analysis["date_fields"] = date_cols
        
        return analysis
    
    def _find_column(self, columns_dict: Dict[str, str], keywords: List[str]) -> Optional[str]:
        """Find column by keywords (case-insensitive)"""
        for keyword in keywords:
            for col_lower, col_original in columns_dict.items():
                if keyword in col_lower:
                    return col_original
        return None
    
    def _find_date_columns(self, csv_data: pd.DataFrame) -> List[str]:
        """Find columns that contain date data"""
        date_columns = []
        
        for col in csv_data.columns:
            # Check column name
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified', 'close']):
                date_columns.append(col)
                continue
            
            # Check data content (sample first few non-null values)
            sample_data = csv_data[col].dropna().head(10)
            if len(sample_data) > 0:
                # Try to parse as date
                try:
                    pd.to_datetime(sample_data.iloc[0])
                    date_columns.append(col)
                except:
                    pass
        
        return date_columns
    
    async def _analyze_opportunity_dates(self, csv_data: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze date-related metrics for opportunities"""
        
        analysis = {}
        current_date = datetime.now()
        
        for date_col in date_cols:
            try:
                dates = pd.to_datetime(csv_data[date_col], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    # Calculate staleness (days since last activity)
                    days_since = (current_date - valid_dates).dt.days
                    
                    analysis[f"{date_col}_analysis"] = {
                        "oldest_date": valid_dates.min().isoformat(),
                        "newest_date": valid_dates.max().isoformat(),
                        "average_days_old": float(days_since.mean()),
                        "stale_records": int((days_since > 30).sum()),  # More than 30 days
                        "very_stale_records": int((days_since > 90).sum())  # More than 90 days
                    }
                    
            except Exception as e:
                logger.warning(f"Could not analyze date column {date_col}: {e}")
        
        return analysis
    
    async def _analyze_activity_timeline(self, csv_data: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze activity timeline"""
        
        analysis = {}
        
        for date_col in date_cols:
            try:
                dates = pd.to_datetime(csv_data[date_col], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    # Group by month
                    monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
                    
                    analysis[f"{date_col}_timeline"] = {
                        "monthly_activity": {str(period): count for period, count in monthly_counts.items()},
                        "total_activities": len(valid_dates),
                        "date_range": {
                            "start": valid_dates.min().isoformat(),
                            "end": valid_dates.max().isoformat()
                        }
                    }
                    
            except Exception as e:
                logger.warning(f"Could not analyze activity timeline for {date_col}: {e}")
        
        return analysis
    
    async def _analyze_revenue_timeline(self, csv_data: pd.DataFrame, date_cols: List[str], revenue_col: str) -> Dict[str, Any]:
        """Analyze revenue over time"""
        
        analysis = {}
        
        for date_col in date_cols:
            try:
                dates = pd.to_datetime(csv_data[date_col], errors='coerce')
                revenue = pd.to_numeric(csv_data[revenue_col], errors='coerce')
                
                # Combine and remove nulls
                combined = pd.DataFrame({'date': dates, 'revenue': revenue}).dropna()
                
                if len(combined) > 0:
                    # Group by month
                    combined['month'] = combined['date'].dt.to_period('M')
                    monthly_revenue = combined.groupby('month')['revenue'].sum()
                    
                    analysis[f"{date_col}_revenue_timeline"] = {
                        "monthly_revenue": {str(period): float(revenue) for period, revenue in monthly_revenue.items()},
                        "total_revenue": float(combined['revenue'].sum()),
                        "average_monthly": float(monthly_revenue.mean()) if len(monthly_revenue) > 0 else 0
                    }
                    
            except Exception as e:
                logger.warning(f"Could not analyze revenue timeline for {date_col}: {e}")
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Data quality recommendations
        if analysis.get("missing_data_percentage", 0) > 20:
            recommendations.append(f"High missing data ({analysis['missing_data_percentage']:.1f}%) - consider data cleanup")
        
        if analysis.get("duplicate_rows", 0) > 0:
            recommendations.append(f"Found {analysis['duplicate_rows']} duplicate records - consider deduplication")
        
        # Opportunity-specific recommendations
        if "stale_records" in str(analysis):
            for key, value in analysis.items():
                if isinstance(value, dict) and "stale_records" in value:
                    if value["stale_records"] > 0:
                        recommendations.append(f"Found {value['stale_records']} stale opportunities requiring attention")
        
        # Health-specific recommendations
        if "health_analysis" in analysis:
            health = analysis["health_analysis"]
            if health.get("at_risk_accounts", 0) > 0:
                recommendations.append(f"Identified {health['at_risk_accounts']} at-risk accounts needing intervention")
        
        # Forecast accuracy recommendations
        if "forecast_accuracy" in analysis:
            accuracy = analysis["forecast_accuracy"]
            if accuracy.get("accuracy_rate", 100) < 80:
                recommendations.append(f"Forecast accuracy is {accuracy['accuracy_rate']:.1f}% - consider improving forecasting process")
        
        if not recommendations:
            recommendations.append("Data analysis completed successfully - review insights for opportunities")
        
        return recommendations
