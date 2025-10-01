"""
CSV Data Loader for Pipeline Agents
Loads opportunity and account data from comprehensive_crm_data.csv
"""

import pandas as pd
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CRMDataLoader:
    """
    CRM Data Loader for CSV-based data
    
    Loads comprehensive CRM data from CSV file including:
    - Opportunities with amounts, probabilities, stages
    - Accounts with industry, revenue data
    - Users (sales reps) with territories
    - Activities and tasks
    """
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # Default path to the comprehensive CRM data
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.csv_path = os.path.join(base_dir, "..", "new folder", "generated_data", "comprehensive_crm_data.csv")
        else:
            self.csv_path = csv_path
            
        self.df = None
        self.opportunities = None
        self.accounts = None
        self.users = None
        self.activities = None
        
        self._load_data()
    
    def _load_data(self) -> bool:
        """Load and parse CSV data"""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                return False
            
            # Load CSV with mixed types handling
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            logger.info(f"✅ Loaded {len(self.df)} records from CSV")
            
            # Separate data by object type
            self.opportunities = self.df[self.df['object'] == 'Opportunity'].copy()
            self.accounts = self.df[self.df['object'] == 'Account'].copy()
            self.users = self.df[self.df['object'] == 'User'].copy()
            self.activities = self.df[self.df['object'] == 'Task'].copy()  # Assuming activities are stored as tasks
            
            logger.info(f"📊 Data breakdown:")
            logger.info(f"   - Opportunities: {len(self.opportunities)}")
            logger.info(f"   - Accounts: {len(self.accounts)}")
            logger.info(f"   - Users: {len(self.users)}")
            logger.info(f"   - Activities: {len(self.activities)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV data: {e}")
            return False
    
    def get_opportunities(self, tenant_id: str = None, user_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get opportunities data formatted for RBA analysis
        
        Args:
            tenant_id: Tenant filter (not used in CSV mode)
            user_id: User filter (optional)
            limit: Maximum number of opportunities to return
            
        Returns:
            List of formatted opportunity dictionaries
        """
        try:
            if self.opportunities is None or len(self.opportunities) == 0:
                logger.warning("No opportunities data available")
                return []
            
            # Get all opportunities (user filtering disabled for now since CSV may not have matching user_id)
            opps = self.opportunities.copy()
            # Note: User filtering disabled - CSV data may not contain the specific user_id
            # if user_id:
            #     opps = opps[opps['OwnerId'] == user_id]
            
            # Apply limit
            opps = opps.head(limit)
            
            # Format for RBA analysis
            formatted_opportunities = []
            
            for _, opp in opps.iterrows():
                # Get related account data
                account_data = self._get_account_by_id(opp.get('AccountId'))
                owner_data = self._get_user_by_id(opp.get('OwnerId'))
                
                formatted_opp = {
                    'Id': opp.get('id'),
                    'Name': opp.get('Name', ''),
                    'Amount': self._safe_float(opp.get('Amount')),
                    'Probability': self._safe_float(opp.get('Probability')),
                    'StageName': opp.get('StageName', ''),
                    'CloseDate': opp.get('CloseDate'),
                    'CreatedDate': opp.get('CreatedDate'),
                    'Type': opp.get('Type', ''),
                    'LeadSource': opp.get('LeadSource', ''),
                    'ActivityDate': opp.get('ActivityDate'),
                    
                    # Account information
                    'Account': {
                        'Id': opp.get('AccountId'),
                        'Name': account_data.get('Name', '') if account_data else '',
                        'Industry': account_data.get('Industry', '') if account_data else '',
                        'AnnualRevenue': self._safe_float(account_data.get('AnnualRevenue')) if account_data else 0,
                        'NumberOfEmployees': self._safe_int(account_data.get('NumberOfEmployees')) if account_data else 0
                    },
                    
                    # Owner information
                    'Owner': {
                        'Id': opp.get('OwnerId'),
                        'Name': f"{owner_data.get('FirstName', '')} {owner_data.get('LastName', '')}".strip() if owner_data else '',
                        'Title': owner_data.get('Title', '') if owner_data else '',
                        'Department': owner_data.get('Department', '') if owner_data else ''
                    },
                    
                    # Additional fields for RBA analysis
                    'DaysSinceCreated': self._calculate_days_since(opp.get('CreatedDate')),
                    'DaysToClose': self._calculate_days_until(opp.get('CloseDate')),
                    'DaysSinceActivity': self._calculate_days_since(opp.get('ActivityDate')),
                    'ExpectedRevenue': self._safe_float(opp.get('Amount')) * (self._safe_float(opp.get('Probability')) / 100.0) if opp.get('Amount') and opp.get('Probability') else 0
                }
                
                formatted_opportunities.append(formatted_opp)
            
            logger.info(f"✅ Formatted {len(formatted_opportunities)} opportunities for RBA analysis")
            return formatted_opportunities
            
        except Exception as e:
            logger.error(f"❌ Failed to get opportunities: {e}")
            return []
    
    def _get_account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account data by ID"""
        if account_id and self.accounts is not None:
            account_rows = self.accounts[self.accounts['id'] == account_id]
            if len(account_rows) > 0:
                return account_rows.iloc[0].to_dict()
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID"""
        if user_id and self.users is not None:
            user_rows = self.users[self.users['id'] == user_id]
            if len(user_rows) > 0:
                return user_rows.iloc[0].to_dict()
        return None
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int"""
        if value is None or pd.isna(value):
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def _calculate_days_since(self, date_str: str) -> int:
        """Calculate days since a given date"""
        if not date_str or pd.isna(date_str):
            return 999  # Unknown/very old
        try:
            if 'T' in str(date_str):
                date_part = str(date_str).split('T')[0]
            else:
                date_part = str(date_str).split(' ')[0]
            
            past_date = datetime.strptime(date_part, '%Y-%m-%d')
            return (datetime.now() - past_date).days
        except:
            return 999
    
    def _calculate_days_until(self, date_str: str) -> int:
        """Calculate days until a given date"""
        if not date_str or pd.isna(date_str):
            return 999  # Unknown/very far
        try:
            if 'T' in str(date_str):
                date_part = str(date_str).split('T')[0]
            else:
                date_part = str(date_str).split(' ')[0]
            
            future_date = datetime.strptime(date_part, '%Y-%m-%d')
            return (future_date - datetime.now()).days
        except:
            return 999
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data"""
        if self.opportunities is None:
            return {"error": "No data loaded"}
        
        try:
            # Calculate key metrics
            total_opportunities = len(self.opportunities)
            total_amount = self.opportunities['Amount'].fillna(0).sum()
            avg_probability = self.opportunities['Probability'].fillna(0).mean()
            
            # Stage distribution
            stage_counts = self.opportunities['StageName'].value_counts().to_dict()
            
            # High-value, low-probability deals (potential sandbagging)
            high_value_threshold = 1000000  # $1M+
            low_prob_threshold = 30  # <30%
            
            sandbagging_candidates = self.opportunities[
                (self.opportunities['Amount'].fillna(0) >= high_value_threshold) &
                (self.opportunities['Probability'].fillna(0) <= low_prob_threshold)
            ]
            
            return {
                "total_opportunities": total_opportunities,
                "total_pipeline_value": total_amount,
                "average_probability": round(avg_probability, 2),
                "stage_distribution": stage_counts,
                "potential_sandbagging_deals": len(sandbagging_candidates),
                "data_quality": {
                    "missing_amounts": len(self.opportunities[self.opportunities['Amount'].isna()]),
                    "missing_probabilities": len(self.opportunities[self.opportunities['Probability'].isna()]),
                    "missing_close_dates": len(self.opportunities[self.opportunities['CloseDate'].isna()])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate summary stats: {e}")
            return {"error": str(e)}

# Global instance
crm_data_loader = CRMDataLoader()
