"""
Microsoft Fabric Service for Salesforce Data Access
Handles real-time Salesforce data queries from Fabric Data Warehouse
"""

import asyncio
import pyodbc
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class FabricQueryResult:
    success: bool
    data: List[Dict]
    row_count: int
    execution_time_ms: float
    error_message: Optional[str] = None

class FabricService:
    """
    Service for accessing Salesforce data through Microsoft Fabric
    Provides real-time access to accounts, opportunities, contacts, and activities
    """
    
    def __init__(self):
        self.connection_string = None
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.timeout_seconds = 30
        
    async def initialize(self):
        """Initialize Fabric connection"""
        try:
            self.connection_string = self.build_connection_string()
            
            # Test connection
            await self.test_connection()
            
            self.logger.info("Fabric service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Fabric service initialization error: {str(e)}")
            raise
    
    def build_connection_string(self) -> str:
        """Build ODBC connection string for Fabric - matches Node.js backend exactly"""
        server = os.getenv('FABRIC_SQL_SERVER')
        database = os.getenv('FABRIC_SQL_DATABASE')
        port = int(os.getenv('FABRIC_SQL_PORT', '1433'))
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        
        if not all([server, database, client_id, client_secret]):
            raise ValueError("Missing required Fabric connection environment variables: FABRIC_SQL_SERVER, FABRIC_SQL_DATABASE, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        
        # Add port if not specified (matches Node.js logic)
        if ',' not in server and ':' not in server:
            server_with_port = f"{server},{port}"
        else:
            server_with_port = server
        
        # Use same SSL settings as Node.js backend
        encrypt = 'yes' if os.getenv('FABRIC_ENCRYPT', 'yes').lower() != 'no' else 'no'
        trust_cert = 'yes' if os.getenv('FABRIC_TRUST_SERVER_CERT', 'false').lower() == 'true' else 'no'
        
        # ODBC Driver 18 with AAD Service Principal auth (same as Node.js)
        connection_string = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={server_with_port};"
            f"DATABASE={database};"
            f"UID={client_id};"
            f"PWD={client_secret};"
            f"Authentication=ActiveDirectoryServicePrincipal;"
            f"Encrypt={encrypt};"
            f"TrustServerCertificate={trust_cert};"
        )
        
        # Debug log (without sensitive info)
        debug_string = connection_string.replace(client_secret, "***")
        self.logger.info(f"🔗 Fabric connection string (Python): {debug_string}")
        
        return connection_string
    
    async def test_connection(self):
        """Test Fabric connection"""
        try:
            result = await self.execute_query("SELECT 1 as test_value")
            if not result.success:
                raise Exception(f"Connection test failed: {result.error_message}")
                
        except Exception as e:
            self.logger.error(f"Fabric connection test failed: {str(e)}")
            raise
    
    async def execute_query(self, query: str, params: List = None) -> FabricQueryResult:
        """Execute SQL query against Fabric with retry logic"""
        start_time = datetime.now()
        
        for attempt in range(self.max_retries):
            try:
                # Execute query in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._execute_sync_query, query, params or []
                )
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return FabricQueryResult(
                    success=True,
                    data=result,
                    row_count=len(result),
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                self.logger.warning(f"Fabric query attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    execution_time = (datetime.now() - start_time).total_seconds() * 1000
                    return FabricQueryResult(
                        success=False,
                        data=[],
                        row_count=0,
                        execution_time_ms=execution_time,
                        error_message=str(e)
                    )
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
        
        return FabricQueryResult(
            success=False,
            data=[],
            row_count=0,
            execution_time_ms=0,
            error_message="Max retries exceeded"
        )
    
    def _execute_sync_query(self, query: str, params: List) -> List[Dict]:
        """Execute query synchronously (called from thread pool)"""
        try:
            with pyodbc.connect(self.connection_string, timeout=self.timeout_seconds) as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch all rows and convert to dictionaries
                rows = cursor.fetchall()
                result = []
                
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Handle different data types
                        if isinstance(value, datetime):
                            row_dict[columns[i]] = value.isoformat()
                        elif value is None:
                            row_dict[columns[i]] = None
                        else:
                            row_dict[columns[i]] = value
                    result.append(row_dict)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Sync query execution error: {str(e)}")
            raise
    
    async def get_account_details(self, account_id: str) -> Dict:
        """Get comprehensive account details from Fabric"""
        try:
            query = """
            SELECT 
                Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
                BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry,
                Phone, Website, Description, OwnerId, 
                CreatedDate, CreatedById, LastModifiedDate, LastModifiedById,
                Rating, AccountSource, ParentId
            FROM dbo.accounts 
            WHERE Id = ?
            """
            
            result = await self.execute_query(query, [account_id])
            
            if result.success and result.data:
                return result.data[0]
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account details: {str(e)}")
            return {}
    
    async def get_account_opportunities(self, account_id: str, include_closed: bool = False) -> List[Dict]:
        """Get opportunities for an account"""
        try:
            query = """
            SELECT 
                Id, Name, AccountId, Description, StageName, Amount, Probability,
                CloseDate, Type, NextStep, LeadSource, IsClosed, IsWon,
                ForecastCategory, OwnerId, CreatedDate, CreatedById,
                LastModifiedDate, LastModifiedById, FiscalQuarter, FiscalYear,
                HasOpportunityLineItem, LastActivityDate, CampaignId, Pricebook2Id
            FROM dbo.opportunities 
            WHERE AccountId = ?
            """
            
            if not include_closed:
                query += " AND IsClosed = 0"
            
            query += " ORDER BY CreatedDate DESC"
            
            result = await self.execute_query(query, [account_id])
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting account opportunities: {str(e)}")
            return []
    
    async def get_account_contacts(self, account_id: str) -> List[Dict]:
        """Get contacts for an account"""
        try:
            query = """
            SELECT 
                Id, FirstName, LastName, Name, Title, Email, Phone, MobilePhone,
                Department, Level__c, AccountId, OwnerId, ReportsToId,
                LeadSource, CreatedDate, LastModifiedDate, LastActivityDate,
                DoNotCall, HasOptedOutOfEmail, EmailBouncedReason
            FROM dbo.contacts 
            WHERE AccountId = ?
            ORDER BY LastName, FirstName
            """
            
            result = await self.execute_query(query, [account_id])
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting account contacts: {str(e)}")
            return []
    
    async def get_opportunity_details(self, opportunity_id: str) -> Dict:
        """Get detailed opportunity information"""
        try:
            query = """
            SELECT 
                o.Id, o.Name, o.AccountId, o.Description, o.StageName, o.Amount, o.Probability,
                o.CloseDate, o.Type, o.NextStep, o.LeadSource, o.IsClosed, o.IsWon,
                o.ForecastCategory, o.OwnerId, o.CreatedDate, o.LastModifiedDate,
                o.FiscalQuarter, o.FiscalYear, o.LastActivityDate,
                a.Name as AccountName, a.Industry as AccountIndustry,
                u.Name as OwnerName, u.Email as OwnerEmail
            FROM dbo.opportunities o
            LEFT JOIN dbo.accounts a ON o.AccountId = a.Id
            LEFT JOIN dbo.users u ON o.OwnerId = u.Id
            WHERE o.Id = ?
            """
            
            result = await self.execute_query(query, [opportunity_id])
            
            if result.success and result.data:
                return result.data[0]
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting opportunity details: {str(e)}")
            return {}
    
    async def get_user_opportunities(self, owner_id: str, stage_filter: str = None) -> List[Dict]:
        """Get opportunities owned by a specific user"""
        try:
            query = """
            SELECT 
                Id, Name, AccountId, StageName, Amount, Probability, CloseDate,
                Type, ForecastCategory, CreatedDate, LastModifiedDate,
                LastActivityDate, IsClosed, IsWon
            FROM dbo.opportunities 
            WHERE OwnerId = ?
            """
            
            params = [owner_id]
            
            if stage_filter:
                query += " AND StageName = ?"
                params.append(stage_filter)
            
            query += " ORDER BY CloseDate ASC"
            
            result = await self.execute_query(query, params)
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting user opportunities: {str(e)}")
            return []
    
    async def get_account_activities(self, account_id: str, days_back: int = 90) -> List[Dict]:
        """Get recent activities for an account"""
        try:
            # Calculate date filter
            date_filter = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                Id, Subject, ActivityDate, ActivityDateTime, WhatId, WhoId,
                Type, Status, Priority, Description, OwnerId,
                CreatedDate, LastModifiedDate, IsTask, IsEvent
            FROM dbo.activities 
            WHERE WhatId = ? AND ActivityDate >= ?
            ORDER BY ActivityDate DESC
            """
            
            result = await self.execute_query(query, [account_id, date_filter])
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting account activities: {str(e)}")
            return []
    
    async def search_accounts(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search accounts by name or other criteria"""
        try:
            query = """
            SELECT TOP (?) 
                Id, Name, Type, Industry, AnnualRevenue, BillingCity, BillingState,
                BillingCountry, OwnerId, CreatedDate, LastModifiedDate
            FROM dbo.accounts 
            WHERE Name LIKE ? OR Industry LIKE ? OR BillingCity LIKE ?
            ORDER BY LastModifiedDate DESC
            """
            
            search_pattern = f"%{search_term}%"
            params = [limit, search_pattern, search_pattern, search_pattern]
            
            result = await self.execute_query(query, params)
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error searching accounts: {str(e)}")
            return []
    
    async def get_pipeline_summary(self, owner_id: str = None, team_ids: List[str] = None) -> Dict:
        """Get pipeline summary data"""
        try:
            query = """
            SELECT 
                StageName,
                COUNT(*) as OpportunityCount,
                SUM(CAST(Amount as DECIMAL(18,2))) as TotalAmount,
                AVG(CAST(Probability as DECIMAL(5,2))) as AvgProbability
            FROM dbo.opportunities 
            WHERE IsClosed = 0
            """
            
            params = []
            
            if owner_id:
                query += " AND OwnerId = ?"
                params.append(owner_id)
            elif team_ids:
                placeholders = ','.join(['?' for _ in team_ids])
                query += f" AND OwnerId IN ({placeholders})"
                params.extend(team_ids)
            
            query += " GROUP BY StageName ORDER BY TotalAmount DESC"
            
            result = await self.execute_query(query, params)
            
            if result.success:
                return {
                    'stages': result.data,
                    'total_pipeline': sum(float(stage.get('TotalAmount', 0) or 0) for stage in result.data),
                    'total_opportunities': sum(int(stage.get('OpportunityCount', 0) or 0) for stage in result.data)
                }
            
            return {'stages': [], 'total_pipeline': 0, 'total_opportunities': 0}
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline summary: {str(e)}")
            return {'stages': [], 'total_pipeline': 0, 'total_opportunities': 0}
    
    async def get_historical_performance(self, owner_id: str, months_back: int = 12) -> Dict:
        """Get historical performance data for revenue analysis"""
        try:
            # Calculate date filter
            date_filter = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                YEAR(CloseDate) as Year,
                MONTH(CloseDate) as Month,
                COUNT(*) as WonOpportunities,
                SUM(CAST(Amount as DECIMAL(18,2))) as WonAmount,
                AVG(CAST(Amount as DECIMAL(18,2))) as AvgDealSize
            FROM dbo.opportunities 
            WHERE OwnerId = ? AND IsWon = 1 AND CloseDate >= ?
            GROUP BY YEAR(CloseDate), MONTH(CloseDate)
            ORDER BY Year DESC, Month DESC
            """
            
            result = await self.execute_query(query, [owner_id, date_filter])
            
            if result.success:
                return {
                    'monthly_performance': result.data,
                    'total_won_amount': sum(float(month.get('WonAmount', 0) or 0) for month in result.data),
                    'total_won_deals': sum(int(month.get('WonOpportunities', 0) or 0) for month in result.data),
                    'avg_deal_size': sum(float(month.get('AvgDealSize', 0) or 0) for month in result.data) / len(result.data) if result.data else 0
                }
            
            return {'monthly_performance': [], 'total_won_amount': 0, 'total_won_deals': 0, 'avg_deal_size': 0}
            
        except Exception as e:
            self.logger.error(f"Error getting historical performance: {str(e)}")
            return {'monthly_performance': [], 'total_won_amount': 0, 'total_won_deals': 0, 'avg_deal_size': 0}
    
    async def get_competitive_intelligence(self, industry: str, limit: int = 100) -> List[Dict]:
        """Get competitive intelligence from account data"""
        try:
            query = """
            SELECT TOP (?)
                Name, Industry, AnnualRevenue, NumberOfEmployees, 
                BillingCountry, Type, CreatedDate,
                COUNT(o.Id) as OpportunityCount,
                SUM(CAST(o.Amount as DECIMAL(18,2))) as TotalPipeline
            FROM dbo.accounts a
            LEFT JOIN dbo.opportunities o ON a.Id = o.AccountId AND o.IsClosed = 0
            WHERE a.Industry = ?
            GROUP BY a.Name, a.Industry, a.AnnualRevenue, a.NumberOfEmployees, 
                     a.BillingCountry, a.Type, a.CreatedDate
            ORDER BY TotalPipeline DESC
            """
            
            result = await self.execute_query(query, [limit, industry])
            
            if result.success:
                return result.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting competitive intelligence: {str(e)}")
            return []
