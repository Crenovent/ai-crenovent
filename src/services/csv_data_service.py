#!/usr/bin/env python3
"""
CSV Data Service
===============

Provides CSV data processing capability as an alternative to Azure Fabric.
Allows users to upload CSV files and query them using SQL-like syntax.
"""

import pandas as pd
import logging
import os
import json
import sqlite3
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

class CSVDataService:
    """Service for handling CSV data uploads and queries"""
    
    def __init__(self):
        self.uploaded_files: Dict[str, pd.DataFrame] = {}
        self.temp_db_path = None
        self.connection = None
        
    async def upload_csv(self, file_content: bytes, filename: str, tenant_id: str) -> Dict[str, Any]:
        """Upload and process CSV file"""
        try:
            # Read CSV content
            csv_content = io.StringIO(file_content.decode('utf-8'))
            df = pd.read_csv(csv_content)
            
            # Store dataframe
            file_key = f"{tenant_id}_{filename}"
            self.uploaded_files[file_key] = df
            
            # Create temporary SQLite database for SQL queries
            await self._create_temp_database(df, filename, tenant_id)
            
            logger.info(f"✅ CSV uploaded successfully: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Clean sample data to handle NaN values for JSON serialization
            sample_df = df.head(3).fillna("")  # Replace NaN with empty strings
            
            return {
                "success": True,
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "sample_data": sample_df.fillna('').to_dict('records'),
                "file_key": file_key
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to upload CSV {filename}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_temp_database(self, df: pd.DataFrame, filename: str, tenant_id: str):
        """Create temporary SQLite database from CSV data"""
        try:
            # Create database file in project data directory
            if not self.temp_db_path:
                # Create data directory if it doesn't exist
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv_uploads")
                os.makedirs(data_dir, exist_ok=True)
                self.temp_db_path = os.path.join(data_dir, f"csv_data_{tenant_id}.db")
                logger.info(f"📂 SQLite database location: {self.temp_db_path}")
            
            # Connect to SQLite
            self.connection = sqlite3.connect(self.temp_db_path)
            
            # Determine table name from filename
            table_name = filename.replace('.csv', '').replace('-', '_').replace(' ', '_').lower()
            
            # Write dataframe to SQLite table
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            
            logger.info(f"✅ Created SQLite table: {table_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create temp database: {e}")
            raise
    
    async def query_csv_data(self, query: str, tenant_id: str) -> Dict[str, Any]:
        """Execute SQL query on uploaded CSV data"""
        try:
            if not self.connection:
                raise Exception("No CSV data uploaded. Please upload a CSV file first.")
            
            # Execute query
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            records = []
            for row in rows:
                record = {}
                for i, value in enumerate(row):
                    # Handle NaN/None values for JSON serialization
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        record[columns[i]] = ""
                    else:
                        record[columns[i]] = value
                records.append(record)
            
            logger.info(f"✅ CSV query executed successfully: {len(records)} records returned")
            
            return {
                "success": True,
                "records": records,
                "row_count": len(records),
                "columns": columns,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"❌ CSV query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def get_available_tables(self, tenant_id: str) -> List[str]:
        """Get list of available tables (CSV files) for tenant"""
        try:
            if not self.connection:
                return []
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            return tables
            
        except Exception as e:
            logger.error(f"❌ Failed to get available tables: {e}")
            return []
    
    async def get_table_schema(self, table_name: str, tenant_id: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        try:
            if not self.connection:
                raise Exception("No database connection")
            
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema_info = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            sample_data = []
            for row in sample_rows:
                record = {}
                for i, value in enumerate(row):
                    # Handle NaN/None values for JSON serialization
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        record[columns[i]] = ""
                    else:
                        record[columns[i]] = value
                sample_data.append(record)
            
            return {
                "table_name": table_name,
                "columns": [{"name": col[1], "type": col[2]} for col in schema_info],
                "sample_data": sample_data,
                "row_count": self._get_row_count(table_name)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get table schema for {table_name}: {e}")
            return {"error": str(e)}
    
    def _get_row_count(self, table_name: str) -> int:
        """Get total row count for table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except:
            return 0
    
    async def cleanup(self):
        """Clean up temporary resources"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            
            if self.temp_db_path and os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
                self.temp_db_path = None
            
            self.uploaded_files.clear()
            logger.info("✅ CSV data service cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Global instance
csv_data_service = CSVDataService()
