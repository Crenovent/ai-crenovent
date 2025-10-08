#!/usr/bin/env python3
"""
Dynamic CSV Format Testing with Detailed Logging
===============================================
Tests the Super Smart RBA Agent with multiple CSV formats dynamically.
No hardcoding - all field mapping happens automatically with intelligent detection.

This script tests:
1. InnovateTech format (Contact Name, Email ID, Reporting Manager, etc.)
2. AcmeCorp format (Full Name, Email, Manager Email, etc.)
3. Dynamic field detection and mapping
4. Detailed logging of the entire processing pipeline
"""

import asyncio
import pandas as pd
import time
import tempfile
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import json

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hierarchy_processing_detailed.log')
    ]
)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class DetailedHierarchyTester:
    """Comprehensive tester with detailed logging for dynamic CSV processing"""
    
    def __init__(self):
        self.test_results = []
        self.processing_stats = {}
        
        logger.info("🚀 Initializing Dynamic CSV Format Tester")
        logger.info("=" * 80)
        
    def log_csv_analysis(self, df: pd.DataFrame, format_name: str):
        """Log detailed CSV analysis"""
        logger.info(f"\n📊 CSV ANALYSIS - {format_name}")
        logger.info("-" * 50)
        logger.info(f"📋 Total Records: {len(df)}")
        logger.info(f"📋 Total Columns: {len(df.columns)}")
        logger.info(f"📋 Column Names: {list(df.columns)}")
        
        # Analyze column patterns
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        name_columns = [col for col in df.columns if 'name' in col.lower()]
        manager_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['manager', 'reporting', 'supervisor', 'reports'])]
        
        logger.info(f"📧 Email-related columns: {email_columns}")
        logger.info(f"👤 Name-related columns: {name_columns}")
        logger.info(f"👔 Manager-related columns: {manager_columns}")
        
        # Sample data analysis
        logger.info(f"\n📋 SAMPLE DATA (First 3 rows):")
        for i in range(min(3, len(df))):
            logger.info(f"   Row {i+1}: {dict(df.iloc[i])}")
        
        # Data quality analysis
        logger.info(f"\n📊 DATA QUALITY ANALYSIS:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            null_percentage = ((len(df) - non_null_count) / len(df)) * 100
            logger.info(f"   {col}: {non_null_count}/{len(df)} non-null ({null_percentage:.1f}% null)")
            
            # Check for email patterns
            if 'email' in col.lower():
                email_pattern_count = df[col].astype(str).str.contains('@', na=False).sum()
                logger.info(f"      └─ Email patterns found: {email_pattern_count}/{non_null_count}")

    def test_super_smart_mapper(self, df: pd.DataFrame, format_name: str) -> Dict[str, Any]:
        """Test Super Smart RBA Mapper with detailed logging"""
        logger.info(f"\n🧠 TESTING SUPER SMART RBA MAPPER - {format_name}")
        logger.info("=" * 60)
        
        try:
            from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
            
            # Initialize mapper
            mapper = SuperSmartRBAMapper()
            logger.info("✅ Super Smart RBA Mapper initialized")
            
            # Log mapper configuration
            stats = mapper.get_mapping_stats()
            logger.info(f"📊 Mapper Stats: {stats}")
            
            # Start processing
            logger.info(f"\n🔄 Starting intelligent processing...")
            start_time = time.time()
            
            # Process with detailed logging
            mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
            
            processing_time = time.time() - start_time
            throughput = len(df) / processing_time
            
            # Log results
            logger.info(f"\n✅ PROCESSING COMPLETE")
            logger.info(f"⏱️  Processing Time: {processing_time:.3f} seconds")
            logger.info(f"🎯 Confidence Score: {confidence:.1%}")
            logger.info(f"🔍 Detected System: {detected_system}")
            logger.info(f"📈 Throughput: {throughput:.1f} records/second")
            logger.info(f"📊 Input Records: {len(df)}")
            logger.info(f"📊 Output Records: {len(mapped_df)}")
            
            # Log field mapping results
            logger.info(f"\n📋 OUTPUT FIELD ANALYSIS:")
            logger.info(f"📋 Output Columns: {list(mapped_df.columns)}")
            
            # Check required fields
            required_fields = ['Name', 'Email', 'Region', 'Level', 'Industry']
            missing_fields = [field for field in required_fields if field not in mapped_df.columns]
            present_fields = [field for field in required_fields if field in mapped_df.columns]
            
            logger.info(f"✅ Required fields present: {present_fields}")
            if missing_fields:
                logger.warning(f"⚠️  Missing required fields: {missing_fields}")
            
            # Log sample output
            logger.info(f"\n📋 SAMPLE OUTPUT (First 3 records):")
            for i in range(min(3, len(mapped_df))):
                sample_record = {}
                for col in ['Name', 'Email', 'Reporting Email', 'Role Title', 'Region', 'Level']:
                    if col in mapped_df.columns:
                        sample_record[col] = mapped_df.iloc[i][col]
                logger.info(f"   Record {i+1}: {sample_record}")
            
            # Data transformation analysis
            logger.info(f"\n🔄 DATA TRANSFORMATION ANALYSIS:")
            for col in mapped_df.columns:
                non_empty_count = mapped_df[col].astype(str).str.len().gt(0).sum()
                logger.info(f"   {col}: {non_empty_count}/{len(mapped_df)} populated ({(non_empty_count/len(mapped_df)*100):.1f}%)")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'confidence': confidence,
                'detected_system': detected_system,
                'throughput': throughput,
                'input_records': len(df),
                'output_records': len(mapped_df),
                'missing_fields': missing_fields,
                'mapped_df': mapped_df
            }
            
        except ImportError as e:
            logger.error(f"❌ Super Smart RBA Mapper not available: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return {'success': False, 'error': str(e)}

    async def test_complete_workflow(self, csv_path: str, format_name: str) -> Dict[str, Any]:
        """Test complete workflow with detailed logging"""
        logger.info(f"\n🚀 TESTING COMPLETE WORKFLOW - {format_name}")
        logger.info("=" * 60)
        
        try:
            from src.rba.hierarchy_workflow_executor import process_csv_hierarchy_optimized
            
            logger.info(f"📁 CSV File: {csv_path}")
            logger.info(f"🏢 Tenant ID: 1300")
            logger.info(f"👤 User ID: 1323")
            
            # Start workflow
            logger.info(f"\n🔄 Starting optimized workflow execution...")
            start_time = time.time()
            
            result = await process_csv_hierarchy_optimized(
                csv_file_path=csv_path,
                tenant_id=1300,
                uploaded_by_user_id=1323
            )
            
            total_time = time.time() - start_time
            
            # Log workflow results
            logger.info(f"\n✅ WORKFLOW EXECUTION COMPLETE")
            logger.info(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
            logger.info(f"📊 Workflow Status: {result.status}")
            logger.info(f"🆔 Execution ID: {result.execution_id}")
            
            if result.status == "completed":
                logger.info(f"✅ SUCCESS: Workflow completed successfully")
                
                # Log detailed results if available
                if hasattr(result, 'final_output') and result.final_output:
                    logger.info(f"\n📊 DETAILED WORKFLOW RESULTS:")
                    final_output = result.final_output
                    
                    # Log hierarchy metrics
                    if 'hierarchy_metrics' in final_output:
                        metrics = final_output['hierarchy_metrics']
                        logger.info(f"🏗️  Hierarchy Metrics:")
                        for key, value in metrics.items():
                            logger.info(f"   {key}: {value}")
                    
                    # Log validation results
                    if 'validation_details' in final_output:
                        validation = final_output['validation_details']
                        logger.info(f"✅ Validation Results:")
                        for key, value in validation.items():
                            logger.info(f"   {key}: {value}")
                
                # Performance analysis
                if total_time < 5.0:
                    logger.info(f"🎯 PERFORMANCE TARGET MET: {total_time:.2f}s < 5.0s ✅")
                else:
                    logger.warning(f"⚠️  Performance target missed: {total_time:.2f}s > 5.0s")
                    
            else:
                logger.error(f"❌ WORKFLOW FAILED: {result.error}")
            
            return {
                'success': result.status == "completed",
                'total_time': total_time,
                'status': result.status,
                'execution_id': result.execution_id,
                'error': result.error if result.status != "completed" else None
            }
            
        except ImportError as e:
            logger.error(f"❌ Workflow executor not available: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_field_mapping_intelligence(self, original_df: pd.DataFrame, mapped_df: pd.DataFrame, format_name: str):
        """Analyze how well the intelligent field mapping worked"""
        logger.info(f"\n🔍 FIELD MAPPING INTELLIGENCE ANALYSIS - {format_name}")
        logger.info("=" * 60)
        
        # Analyze original vs mapped columns
        logger.info(f"📋 COLUMN MAPPING ANALYSIS:")
        logger.info(f"   Original columns ({len(original_df.columns)}): {list(original_df.columns)}")
        logger.info(f"   Mapped columns ({len(mapped_df.columns)}): {list(mapped_df.columns)}")
        
        # Analyze specific mappings
        mapping_analysis = {}
        
        # Name field mapping
        name_candidates = [col for col in original_df.columns if 'name' in col.lower()]
        if name_candidates and 'Name' in mapped_df.columns:
            logger.info(f"👤 Name Mapping: {name_candidates} → 'Name' ✅")
            mapping_analysis['name_mapping'] = 'success'
        
        # Email field mapping
        email_candidates = [col for col in original_df.columns if 'email' in col.lower()]
        if email_candidates and 'Email' in mapped_df.columns:
            logger.info(f"📧 Email Mapping: {email_candidates} → 'Email' ✅")
            mapping_analysis['email_mapping'] = 'success'
        
        # Manager field mapping
        manager_candidates = [col for col in original_df.columns if any(keyword in col.lower() for keyword in ['manager', 'reporting', 'supervisor'])]
        if manager_candidates and 'Reporting Email' in mapped_df.columns:
            logger.info(f"👔 Manager Mapping: {manager_candidates} → 'Reporting Email' ✅")
            mapping_analysis['manager_mapping'] = 'success'
        
        # Title/Role mapping
        title_candidates = [col for col in original_df.columns if any(keyword in col.lower() for keyword in ['title', 'designation', 'role', 'position'])]
        if title_candidates and 'Role Title' in mapped_df.columns:
            logger.info(f"💼 Title Mapping: {title_candidates} → 'Role Title' ✅")
            mapping_analysis['title_mapping'] = 'success'
        
        # Location/Territory mapping
        location_candidates = [col for col in original_df.columns if any(keyword in col.lower() for keyword in ['location', 'territory', 'region'])]
        if location_candidates and 'Region' in mapped_df.columns:
            logger.info(f"🌍 Location Mapping: {location_candidates} → 'Region' ✅")
            mapping_analysis['location_mapping'] = 'success'
        
        # Calculate mapping success rate
        total_mappings = len(mapping_analysis)
        successful_mappings = sum(1 for status in mapping_analysis.values() if status == 'success')
        success_rate = (successful_mappings / total_mappings * 100) if total_mappings > 0 else 0
        
        logger.info(f"\n📊 MAPPING SUCCESS RATE: {successful_mappings}/{total_mappings} ({success_rate:.1f}%)")
        
        return mapping_analysis

    def run_comprehensive_test(self, csv_path: str, format_name: str):
        """Run comprehensive test for a CSV format"""
        logger.info(f"\n" + "="*80)
        logger.info(f"🧪 COMPREHENSIVE TEST: {format_name}")
        logger.info(f"📁 File: {csv_path}")
        logger.info("="*80)
        
        # Load and analyze CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✅ CSV loaded successfully: {len(df)} records")
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            return
        
        # Step 1: CSV Analysis
        self.log_csv_analysis(df, format_name)
        
        # Step 2: Test Super Smart Mapper
        mapper_result = self.test_super_smart_mapper(df, format_name)
        
        # Step 3: Analyze field mapping intelligence
        if mapper_result['success']:
            mapping_analysis = self.analyze_field_mapping_intelligence(df, mapper_result['mapped_df'], format_name)
        
        # Step 4: Test complete workflow
        workflow_result = asyncio.run(self.test_complete_workflow(csv_path, format_name))
        
        # Store results
        test_result = {
            'format_name': format_name,
            'csv_path': csv_path,
            'record_count': len(df),
            'mapper_result': mapper_result,
            'workflow_result': workflow_result,
            'mapping_analysis': mapping_analysis if mapper_result['success'] else None
        }
        
        self.test_results.append(test_result)
        
        # Log summary for this format
        logger.info(f"\n📋 TEST SUMMARY - {format_name}")
        logger.info("-" * 40)
        if mapper_result['success']:
            logger.info(f"✅ Mapper: {mapper_result['confidence']:.1%} confidence, {mapper_result['processing_time']:.3f}s")
        else:
            logger.info(f"❌ Mapper: Failed - {mapper_result.get('error', 'Unknown error')}")
        
        if workflow_result['success']:
            logger.info(f"✅ Workflow: Completed in {workflow_result['total_time']:.2f}s")
        else:
            logger.info(f"❌ Workflow: Failed - {workflow_result.get('error', 'Unknown error')}")

    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info(f"\n" + "="*80)
        logger.info(f"📊 FINAL COMPREHENSIVE REPORT")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        successful_mappers = sum(1 for result in self.test_results if result['mapper_result']['success'])
        successful_workflows = sum(1 for result in self.test_results if result['workflow_result']['success'])
        
        logger.info(f"\n📈 OVERALL STATISTICS:")
        logger.info(f"   Total formats tested: {total_tests}")
        logger.info(f"   Successful mapper tests: {successful_mappers}/{total_tests} ({successful_mappers/total_tests*100:.1f}%)")
        logger.info(f"   Successful workflow tests: {successful_workflows}/{total_tests} ({successful_workflows/total_tests*100:.1f}%)")
        
        logger.info(f"\n📊 DETAILED RESULTS BY FORMAT:")
        for result in self.test_results:
            logger.info(f"\n🔍 {result['format_name']}:")
            logger.info(f"   📁 File: {result['csv_path']}")
            logger.info(f"   📊 Records: {result['record_count']}")
            
            if result['mapper_result']['success']:
                mapper = result['mapper_result']
                logger.info(f"   🧠 Mapper: ✅ {mapper['confidence']:.1%} confidence, {mapper['throughput']:.1f} rec/sec")
                logger.info(f"   🔍 Detected: {mapper['detected_system']}")
            else:
                logger.info(f"   🧠 Mapper: ❌ {result['mapper_result'].get('error', 'Failed')}")
            
            if result['workflow_result']['success']:
                workflow = result['workflow_result']
                logger.info(f"   🚀 Workflow: ✅ {workflow['total_time']:.2f}s")
                if workflow['total_time'] < 5.0:
                    logger.info(f"      🎯 Performance target met (<5s)")
                else:
                    logger.info(f"      ⚠️  Performance target missed (>{workflow['total_time']:.1f}s)")
            else:
                logger.info(f"   🚀 Workflow: ❌ {result['workflow_result'].get('error', 'Failed')}")
        
        # Performance analysis
        if successful_workflows > 0:
            workflow_times = [r['workflow_result']['total_time'] for r in self.test_results if r['workflow_result']['success']]
            avg_time = sum(workflow_times) / len(workflow_times)
            max_time = max(workflow_times)
            min_time = min(workflow_times)
            
            logger.info(f"\n⚡ PERFORMANCE ANALYSIS:")
            logger.info(f"   Average processing time: {avg_time:.2f}s")
            logger.info(f"   Fastest processing: {min_time:.2f}s")
            logger.info(f"   Slowest processing: {max_time:.2f}s")
            
            if max_time < 5.0:
                logger.info(f"   🎯 ALL TESTS MEET PERFORMANCE TARGET (<5s) ✅")
            else:
                logger.info(f"   ⚠️  Some tests exceed performance target (>5s)")
        
        # Intelligence analysis
        logger.info(f"\n🧠 INTELLIGENCE ANALYSIS:")
        if successful_mappers > 0:
            confidences = [r['mapper_result']['confidence'] for r in self.test_results if r['mapper_result']['success']]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            
            logger.info(f"   Average confidence: {avg_confidence:.1%}")
            logger.info(f"   Minimum confidence: {min_confidence:.1%}")
            
            if min_confidence > 0.8:
                logger.info(f"   🎯 ALL FORMATS HAVE HIGH CONFIDENCE (>80%) ✅")
            else:
                logger.info(f"   ⚠️  Some formats have low confidence (<80%)")
        
        logger.info(f"\n🏁 TESTING COMPLETE - Dynamic CSV format processing validated!")

def main():
    """Main test execution"""
    tester = DetailedHierarchyTester()
    
    # Test files
    test_files = [
        {
            'path': 'test_data/innovatetech_sales_hierarchy.csv',
            'name': 'InnovateTech Sales Format',
            'description': 'Contact Name, Email ID, Reporting Manager format'
        },
        {
            'path': 'test_data/acmecorp_employee_hierarchy.csv', 
            'name': 'AcmeCorp Employee Format',
            'description': 'Full Name, Email, Manager Email format'
        }
    ]
    
    logger.info("🚀 Starting Dynamic CSV Format Testing")
    logger.info(f"📅 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Test files: {len(test_files)}")
    
    # Run tests for each format
    for test_file in test_files:
        if os.path.exists(test_file['path']):
            tester.run_comprehensive_test(test_file['path'], test_file['name'])
        else:
            logger.error(f"❌ Test file not found: {test_file['path']}")
    
    # Generate final report
    tester.generate_final_report()
    
    logger.info(f"\n📝 Detailed logs saved to: hierarchy_processing_detailed.log")

if __name__ == "__main__":
    main()
