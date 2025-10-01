"""
CSV LLM Processor - Fallback for RBA Hierarchy Processing
========================================================
This serves as an intelligent fallback when the primary RBA DSL workflow fails.
Uses LLM to understand and process CSV structures that might not be handled by the rule-based system.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import openai
import os

logger = logging.getLogger(__name__)

class CSVLLMProcessor:
    """
    LLM-powered CSV processor for complex hierarchy structures.
    
    This is a fallback processor that uses Large Language Models to:
    1. Understand complex CSV structures
    2. Map fields intelligently when rules fail
    3. Resolve ambiguous reporting relationships
    4. Handle edge cases that rule-based systems miss
    """
    
    def __init__(self):
        self.client = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM client (OpenAI or Azure OpenAI)"""
        try:
            # Try OpenAI first
            openai_key = os.getenv('OPENAI_API_KEY')
            azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
            
            if openai_key:
                openai.api_key = openai_key
                self.client = openai
                logger.info("✅ LLM Processor initialized with OpenAI")
            elif azure_openai_key:
                # Configure for Azure OpenAI
                openai.api_key = azure_openai_key
                openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT', 'https://your-resource.openai.azure.com/')
                openai.api_type = "azure"
                openai.api_version = "2024-02-15-preview"
                self.client = openai
                logger.info("✅ LLM Processor initialized with Azure OpenAI")
            else:
                logger.warning("⚠️ No OpenAI API key found (checked OPENAI_API_KEY and AZURE_OPENAI_API_KEY) - LLM fallback disabled")
                self.client = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}")
            self.client = None
    
    async def process_csv_with_llm_fallback(
        self, 
        df: pd.DataFrame, 
        tenant_id: int,
        processing_context: Dict[str, Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process CSV using LLM when RBA processing fails.
        
        Args:
            df: Input DataFrame
            tenant_id: Tenant ID for context
            processing_context: Context from failed RBA processing
            
        Returns:
            Tuple of (processed_dataframe, processing_summary)
        """
        logger.info(f"🧠 [LLM Fallback] Processing CSV with {len(df)} rows for tenant {tenant_id}")
        
        if not self.client:
            raise ValueError("LLM client not available - cannot process as fallback")
        
        try:
            # Step 1: Analyze CSV structure with LLM
            structure_analysis = await self._analyze_csv_structure(df)
            
            # Step 2: Map fields intelligently
            field_mapping = await self._map_fields_with_llm(df, structure_analysis)
            
            # Step 3: Build hierarchy relationships
            hierarchy_analysis = await self._analyze_hierarchy_relationships(df, field_mapping)
            
            # Step 4: Normalize data
            normalized_df = await self._normalize_data_with_llm(df, field_mapping, hierarchy_analysis)
            
            processing_summary = {
                "processor_type": "LLM_FALLBACK",
                "structure_analysis": structure_analysis,
                "field_mapping": field_mapping,
                "hierarchy_analysis": hierarchy_analysis,
                "confidence_score": structure_analysis.get("confidence", 0.7),
                "processed_records": len(normalized_df),
                "processing_warnings": []
            }
            
            logger.info(f"✅ [LLM Fallback] Successfully processed {len(normalized_df)} records")
            return normalized_df, processing_summary
        
        except Exception as e:
            logger.error(f"❌ [LLM Fallback] Processing failed: {e}")
        raise
    
    async def _analyze_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM to analyze CSV structure and identify patterns"""
        
        # Sample first few rows for analysis
        sample_data = df.head(5).to_string()
        column_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns}
        }
        
        prompt = f"""
        Analyze this CSV structure and identify the organizational hierarchy fields:
        
        Columns: {column_info['columns']}
        Sample Data:
        {sample_data}
        
        Please identify:
        1. Employee name field
        2. Email field
        3. Manager/Reports-to field (name or email)
        4. Job title/role field
        5. Department field
        6. Any hierarchy level indicators
        7. Confidence level (0-1) in your analysis
        
        Respond in JSON format:
        {{
            "name_field": "column_name",
            "email_field": "column_name", 
            "manager_field": "column_name",
            "manager_field_type": "email|name",
            "title_field": "column_name",
            "department_field": "column_name",
            "level_field": "column_name",
            "confidence": 0.95,
            "analysis_notes": "explanation"
        }}
        """
        
        try:
            if self.client:
                response = await self.client.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing organizational data structures."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                logger.info(f"🧠 LLM Structure Analysis: {result.get('confidence', 0)} confidence")
                return result
            else:
                # Fallback heuristic analysis
                return self._heuristic_structure_analysis(df)
                
        except Exception as e:
            logger.error(f"❌ LLM structure analysis failed: {e}")
            return self._heuristic_structure_analysis(df)
    
    def _heuristic_structure_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback heuristic analysis when LLM is not available"""
        
        columns = [col.lower() for col in df.columns]
        
        # Heuristic field detection
        name_field = None
        email_field = None
        manager_field = None
        title_field = None
        
        for i, col in enumerate(columns):
            original_col = df.columns[i]
            
            if not name_field and any(keyword in col for keyword in ['name', 'employee', 'person']):
                name_field = original_col
            
            if not email_field and 'email' in col:
                email_field = original_col
                
            if not manager_field and any(keyword in col for keyword in ['manager', 'reports', 'supervisor', 'boss']):
                manager_field = original_col
                
            if not title_field and any(keyword in col for keyword in ['title', 'role', 'position', 'job']):
                title_field = original_col
        
        return {
            "name_field": name_field,
            "email_field": email_field,
            "manager_field": manager_field,
            "manager_field_type": "email" if manager_field and 'email' in manager_field.lower() else "name",
            "title_field": title_field,
            "department_field": None,
            "level_field": None,
            "confidence": 0.6,
            "analysis_notes": "Heuristic analysis fallback"
        }
    
    async def _map_fields_with_llm(self, df: pd.DataFrame, structure_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Map CSV fields to Crenovent standard format"""
        
        field_mapping = {}
        
        # Map identified fields to standard format
        if structure_analysis.get("name_field"):
            field_mapping["Name"] = structure_analysis["name_field"]
            
        if structure_analysis.get("email_field"):
            field_mapping["Email"] = structure_analysis["email_field"]
            
        if structure_analysis.get("manager_field"):
            if structure_analysis.get("manager_field_type") == "email":
                field_mapping["Reports To (Email)"] = structure_analysis["manager_field"]
        else:
                field_mapping["Reporting Manager Name"] = structure_analysis["manager_field"]
                
        if structure_analysis.get("title_field"):
            field_mapping["Role Title"] = structure_analysis["title_field"]
            
        if structure_analysis.get("department_field"):
            field_mapping["Department"] = structure_analysis["department_field"]
            
        if structure_analysis.get("level_field"):
            field_mapping["Level"] = structure_analysis["level_field"]
        
        logger.info(f"🔄 [LLM] Field mapping: {field_mapping}")
        return field_mapping
    
    async def _analyze_hierarchy_relationships(self, df: pd.DataFrame, field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Analyze hierarchy relationships in the data"""
        
        hierarchy_stats = {
            "total_employees": len(df),
            "employees_with_managers": 0,
            "potential_root_nodes": 0,
            "hierarchy_depth_estimate": 1
        }
        
        # Count employees with manager information
        manager_field = field_mapping.get("Reports To (Email)") or field_mapping.get("Reporting Manager Name")
        if manager_field and manager_field in df.columns:
            non_null_managers = df[manager_field].notna().sum()
            hierarchy_stats["employees_with_managers"] = int(non_null_managers)
            hierarchy_stats["potential_root_nodes"] = len(df) - int(non_null_managers)
        
        # Estimate hierarchy depth (simple heuristic)
        if hierarchy_stats["employees_with_managers"] > 0:
            hierarchy_stats["hierarchy_depth_estimate"] = min(5, max(1, int(len(df) / 10)))
        
        return hierarchy_stats
    
    async def _normalize_data_with_llm(
        self, 
        df: pd.DataFrame, 
        field_mapping: Dict[str, str],
        hierarchy_analysis: Dict[str, Any]
    ) -> pd.DataFrame:
        """Normalize the data to Crenovent standard format"""
        
        normalized_df = pd.DataFrame()
        
        # Map fields according to the field mapping
        for standard_field, csv_field in field_mapping.items():
            if csv_field in df.columns:
                normalized_df[standard_field] = df[csv_field]
        
        # Add default values for missing required fields
        if "Name" not in normalized_df.columns:
            normalized_df["Name"] = df.iloc[:, 0]  # Use first column as name fallback
            
        if "Email" not in normalized_df.columns:
            # Generate placeholder emails if missing
            normalized_df["Email"] = [f"user{i}@company.com" for i in range(len(df))]
            
        # Add standard Crenovent fields
        normalized_df["Industry"] = "Sass"  # Default industry
        normalized_df["Region"] = "America"  # Default region
        normalized_df["Segment"] = "Enterprise"  # Default segment
        normalized_df["Business Function"] = "Sales"  # Default function
        normalized_df["Role Function"] = "Sales"  # Default role function
        
        # Infer levels if not present
        if "Level" not in normalized_df.columns:
            normalized_df["Level"] = self._infer_levels_from_hierarchy(normalized_df, hierarchy_analysis)
        
        # Add modules based on role
        normalized_df["Modules"] = "Forecasting,Planning,Pipeline"
        
        logger.info(f"📊 [LLM] Normalized {len(normalized_df)} records with {len(normalized_df.columns)} fields")
        return normalized_df
    
    def _infer_levels_from_hierarchy(self, df: pd.DataFrame, hierarchy_analysis: Dict[str, Any]) -> List[str]:
        """Infer hierarchy levels based on reporting relationships"""
        
        levels = []
        total_employees = len(df)
        
        # Simple level inference based on position in the hierarchy
        for i in range(total_employees):
            # Estimate level based on hierarchy statistics
            if hierarchy_analysis.get("potential_root_nodes", 0) > 0 and i < hierarchy_analysis["potential_root_nodes"]:
                # Likely a senior role
                levels.append("M6")
            elif i < total_employees * 0.1:
                # Top 10% - likely managers
                levels.append("M5")
            elif i < total_employees * 0.3:
                # Next 20% - likely senior individual contributors
                levels.append("M4")
            else:
                # Rest - individual contributors
                levels.append("IC")
        
        return levels
    
    def is_available(self) -> bool:
        """Check if LLM processor is available"""
        return self.client is not None
    
    async def validate_processing_quality(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of LLM processing"""
        
        validation_results = {
            "data_preservation": len(processed_df) / len(original_df) if len(original_df) > 0 else 0,
            "required_fields_present": all(field in processed_df.columns for field in ["Name", "Email"]),
            "hierarchy_fields_mapped": "Reports To (Email)" in processed_df.columns or "Reporting Manager Name" in processed_df.columns,
            "quality_score": 0.0
        }
        
        # Calculate overall quality score
        quality_factors = [
            validation_results["data_preservation"],
            1.0 if validation_results["required_fields_present"] else 0.0,
            1.0 if validation_results["hierarchy_fields_mapped"] else 0.5
        ]
        
        validation_results["quality_score"] = sum(quality_factors) / len(quality_factors)
        
        logger.info(f"🎯 [LLM] Processing quality score: {validation_results['quality_score']:.2f}")
        return validation_results

# Factory function
def create_llm_processor() -> CSVLLMProcessor:
    """Create LLM processor instance"""
    return CSVLLMProcessor()

# Export for use in RBA workflow
__all__ = ["CSVLLMProcessor", "create_llm_processor"]
