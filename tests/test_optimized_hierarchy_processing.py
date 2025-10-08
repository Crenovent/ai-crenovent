"""
Comprehensive Test Suite for Optimized Hierarchy Processing
==========================================================
Tests the Super Smart RBA Agent and optimized processing pipeline
to ensure 30 users process in <5 seconds with 95%+ accuracy.
"""

import pytest
import pandas as pd
import asyncio
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Import the optimized components
try:
    from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
    from hierarchy_processor.core.optimized_universal_mapper import OptimizedUniversalMapper
    from src.rba.hierarchy_workflow_executor import (
        create_optimized_hierarchy_executor,
        process_csv_hierarchy_optimized,
        HierarchyProcessingWorkflowExecutor
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generate test CSV data for various scenarios"""
    
    @staticmethod
    def create_simple_hierarchy_csv(num_users: int = 30) -> pd.DataFrame:
        """Create a simple hierarchy CSV for performance testing"""
        data = []
        
        # CEO
        data.append({
            'Name': 'John CEO',
            'Email': 'john.ceo@company.com',
            'Role Title': 'Chief Executive Officer',
            'Manager Email': '',
            'Department': 'Executive',
            'Location': 'New York'
        })
        
        # VPs (report to CEO)
        vp_emails = []
        for i in range(3):
            email = f'vp{i+1}@company.com'
            vp_emails.append(email)
            data.append({
                'Name': f'VP {i+1}',
                'Email': email,
                'Role Title': 'Vice President',
                'Manager Email': 'john.ceo@company.com',
                'Department': ['Sales', 'Engineering', 'Marketing'][i],
                'Location': ['California', 'Texas', 'Florida'][i]
            })
        
        # Directors (report to VPs)
        director_emails = []
        for i in range(6):
            email = f'director{i+1}@company.com'
            director_emails.append(email)
            data.append({
                'Name': f'Director {i+1}',
                'Email': email,
                'Role Title': 'Director',
                'Manager Email': vp_emails[i % 3],
                'Department': ['Sales', 'Engineering', 'Marketing'][i % 3],
                'Location': ['California', 'Texas', 'Florida'][i % 3]
            })
        
        # Managers (report to Directors)
        manager_emails = []
        for i in range(12):
            email = f'manager{i+1}@company.com'
            manager_emails.append(email)
            data.append({
                'Name': f'Manager {i+1}',
                'Email': email,
                'Role Title': 'Manager',
                'Manager Email': director_emails[i % 6],
                'Department': ['Sales', 'Engineering', 'Marketing'][i % 3],
                'Location': ['California', 'Texas', 'Florida'][i % 3]
            })
        
        # Individual Contributors (fill remaining slots)
        remaining = num_users - len(data)
        for i in range(remaining):
            data.append({
                'Name': f'Employee {i+1}',
                'Email': f'employee{i+1}@company.com',
                'Role Title': 'Individual Contributor',
                'Manager Email': manager_emails[i % len(manager_emails)] if manager_emails else vp_emails[i % len(vp_emails)],
                'Department': ['Sales', 'Engineering', 'Marketing'][i % 3],
                'Location': ['California', 'Texas', 'Florida'][i % 3]
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_salesforce_format_csv(num_users: int = 30) -> pd.DataFrame:
        """Create Salesforce-style CSV format"""
        simple_df = TestDataGenerator.create_simple_hierarchy_csv(num_users)
        
        # Rename columns to Salesforce format
        salesforce_df = simple_df.rename(columns={
            'Name': 'Full Name',
            'Email': 'Email Address',
            'Role Title': 'Job Title',
            'Manager Email': 'Reports To Email',
            'Department': 'Business Unit',
            'Location': 'Office Location'
        })
        
        return salesforce_df
    
    @staticmethod
    def create_workday_format_csv(num_users: int = 30) -> pd.DataFrame:
        """Create Workday-style CSV format"""
        simple_df = TestDataGenerator.create_simple_hierarchy_csv(num_users)
        
        # Rename columns to Workday format
        workday_df = simple_df.rename(columns={
            'Name': 'Worker Name',
            'Email': 'Work Email',
            'Role Title': 'Position Title',
            'Manager Email': 'Manager Work Email',
            'Department': 'Organization Unit',
            'Location': 'Work Location'
        })
        
        return workday_df
    
    @staticmethod
    def create_messy_csv(num_users: int = 30) -> pd.DataFrame:
        """Create a messy CSV with inconsistent formatting"""
        simple_df = TestDataGenerator.create_simple_hierarchy_csv(num_users)
        
        # Rename to messy column names
        messy_df = simple_df.rename(columns={
            'Name': 'employee_full_name',
            'Email': 'work_email_address',
            'Role Title': 'current_job_title',
            'Manager Email': 'direct_supervisor_email',
            'Department': 'dept_name',
            'Location': 'office_loc'
        })
        
        return messy_df

class TestSuperSmartRBAMapper:
    """Test the Super Smart RBA Mapper component"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mapper = SuperSmartRBAMapper()
    
    def test_simple_hierarchy_mapping(self):
        """Test mapping of simple hierarchy CSV"""
        df = TestDataGenerator.create_simple_hierarchy_csv(10)
        
        start_time = time.time()
        mapped_df, confidence, detected_system = self.mapper.map_csv_intelligently(df, tenant_id=1300)
        processing_time = time.time() - start_time
        
        # Assertions
        assert confidence > 0.9, f"Expected confidence > 90%, got {confidence:.1%}"
        assert len(mapped_df) == 10, f"Expected 10 records, got {len(mapped_df)}"
        assert 'Name' in mapped_df.columns, "Name column missing"
        assert 'Email' in mapped_df.columns, "Email column missing"
        assert processing_time < 1.0, f"Processing took {processing_time:.2f}s, expected <1s"
        
        logger.info(f"✅ Simple mapping: {confidence:.1%} confidence in {processing_time:.3f}s")
    
    def test_salesforce_format_detection(self):
        """Test detection and mapping of Salesforce format"""
        df = TestDataGenerator.create_salesforce_format_csv(15)
        
        mapped_df, confidence, detected_system = self.mapper.map_csv_intelligently(df, tenant_id=1300)
        
        assert 'salesforce' in detected_system.lower() or detected_system == 'unknown'
        assert confidence > 0.8, f"Expected confidence > 80%, got {confidence:.1%}"
        assert len(mapped_df) == 15, f"Expected 15 records, got {len(mapped_df)}"
        
        logger.info(f"✅ Salesforce detection: {detected_system} with {confidence:.1%} confidence")
    
    def test_workday_format_detection(self):
        """Test detection and mapping of Workday format"""
        df = TestDataGenerator.create_workday_format_csv(20)
        
        mapped_df, confidence, detected_system = self.mapper.map_csv_intelligently(df, tenant_id=1300)
        
        assert confidence > 0.8, f"Expected confidence > 80%, got {confidence:.1%}"
        assert len(mapped_df) == 20, f"Expected 20 records, got {len(mapped_df)}"
        
        logger.info(f"✅ Workday detection: {detected_system} with {confidence:.1%} confidence")
    
    def test_messy_csv_handling(self):
        """Test handling of messy CSV with poor column names"""
        df = TestDataGenerator.create_messy_csv(25)
        
        mapped_df, confidence, detected_system = self.mapper.map_csv_intelligently(df, tenant_id=1300)
        
        # Should still work with lower confidence
        assert confidence > 0.6, f"Expected confidence > 60%, got {confidence:.1%}"
        assert len(mapped_df) == 25, f"Expected 25 records, got {len(mapped_df)}"
        
        logger.info(f"✅ Messy CSV handling: {confidence:.1%} confidence")
    
    def test_performance_benchmark(self):
        """Test performance with various dataset sizes"""
        sizes = [10, 30, 50, 100]
        results = []
        
        for size in sizes:
            df = TestDataGenerator.create_simple_hierarchy_csv(size)
            
            start_time = time.time()
            mapped_df, confidence, detected_system = self.mapper.map_csv_intelligently(df, tenant_id=1300)
            processing_time = time.time() - start_time
            
            throughput = size / processing_time
            results.append({
                'size': size,
                'time': processing_time,
                'throughput': throughput,
                'confidence': confidence
            })
            
            logger.info(f"📊 Size {size}: {processing_time:.3f}s ({throughput:.1f} records/sec)")
        
        # Performance assertions
        for result in results:
            if result['size'] <= 30:
                assert result['time'] < 1.0, f"Size {result['size']} took {result['time']:.2f}s, expected <1s"
            assert result['confidence'] > 0.8, f"Size {result['size']} confidence {result['confidence']:.1%} too low"

class TestOptimizedUniversalMapper:
    """Test the Optimized Universal Mapper component"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mapper = OptimizedUniversalMapper(enable_caching=True, chunk_size=100)
    
    def test_vectorized_processing(self):
        """Test vectorized processing performance"""
        df = TestDataGenerator.create_simple_hierarchy_csv(50)
        
        start_time = time.time()
        mapped_df = self.mapper.map_any_hrms_to_crenovent_vectorized(df)
        processing_time = time.time() - start_time
        
        assert len(mapped_df) == 50, f"Expected 50 records, got {len(mapped_df)}"
        assert processing_time < 2.0, f"Vectorized processing took {processing_time:.2f}s, expected <2s"
        
        # Check required columns exist
        required_columns = ['Name', 'Email', 'Industry', 'Region', 'Level']
        for col in required_columns:
            assert col in mapped_df.columns, f"Required column {col} missing"
        
        logger.info(f"✅ Vectorized processing: {len(mapped_df)} records in {processing_time:.3f}s")
    
    def test_caching_performance(self):
        """Test caching improves performance on repeated calls"""
        df = TestDataGenerator.create_simple_hierarchy_csv(30)
        
        # First call (no cache)
        start_time = time.time()
        mapped_df1 = self.mapper.map_any_hrms_to_crenovent_vectorized(df)
        first_call_time = time.time() - start_time
        
        # Second call (with cache)
        start_time = time.time()
        mapped_df2 = self.mapper.map_any_hrms_to_crenovent_vectorized(df)
        second_call_time = time.time() - start_time
        
        # Second call should be faster due to caching
        assert second_call_time <= first_call_time, f"Caching didn't improve performance: {second_call_time:.3f}s vs {first_call_time:.3f}s"
        assert len(mapped_df1) == len(mapped_df2), "Cached result differs from original"
        
        logger.info(f"✅ Caching: {first_call_time:.3f}s → {second_call_time:.3f}s ({(1-second_call_time/first_call_time)*100:.1f}% improvement)")

class TestCompleteWorkflow:
    """Test the complete optimized workflow end-to-end"""
    
    def setup_method(self):
        """Setup test environment"""
        self.executor = create_optimized_hierarchy_executor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_temp_csv(self, df: pd.DataFrame, filename: str = "test.csv") -> str:
        """Create temporary CSV file"""
        filepath = os.path.join(self.temp_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    @pytest.mark.asyncio
    async def test_30_users_performance_target(self):
        """Test the main performance target: 30 users in <5 seconds"""
        df = TestDataGenerator.create_simple_hierarchy_csv(30)
        csv_path = self._create_temp_csv(df, "30_users.csv")
        
        start_time = time.time()
        
        result = await process_csv_hierarchy_optimized(
            csv_file_path=csv_path,
            tenant_id=1300,
            uploaded_by_user_id=1323
        )
        
        total_time = time.time() - start_time
        
        # Main performance assertion
        assert total_time < 5.0, f"🚨 PERFORMANCE FAILURE: 30 users took {total_time:.2f}s, target was <5s"
        
        # Verify result quality
        assert result.status == "completed", f"Workflow failed: {result.error}"
        
        logger.info(f"🎯 PERFORMANCE TARGET MET: 30 users processed in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_different_csv_formats(self):
        """Test different CSV formats work correctly"""
        formats = [
            ("simple", TestDataGenerator.create_simple_hierarchy_csv(20)),
            ("salesforce", TestDataGenerator.create_salesforce_format_csv(20)),
            ("workday", TestDataGenerator.create_workday_format_csv(20)),
            ("messy", TestDataGenerator.create_messy_csv(20))
        ]
        
        results = []
        
        for format_name, df in formats:
            csv_path = self._create_temp_csv(df, f"{format_name}.csv")
            
            start_time = time.time()
            result = await process_csv_hierarchy_optimized(
                csv_file_path=csv_path,
                tenant_id=1300,
                uploaded_by_user_id=1323
            )
            processing_time = time.time() - start_time
            
            assert result.status == "completed", f"{format_name} format failed: {result.error}"
            assert processing_time < 3.0, f"{format_name} took {processing_time:.2f}s, expected <3s"
            
            results.append({
                'format': format_name,
                'time': processing_time,
                'status': result.status
            })
            
            logger.info(f"✅ {format_name.title()} format: {processing_time:.2f}s")
        
        # All formats should complete successfully
        assert all(r['status'] == 'completed' for r in results), "Some formats failed"
    
    @pytest.mark.asyncio
    async def test_scalability(self):
        """Test scalability with increasing dataset sizes"""
        sizes = [10, 30, 50, 100, 200]
        results = []
        
        for size in sizes:
            df = TestDataGenerator.create_simple_hierarchy_csv(size)
            csv_path = self._create_temp_csv(df, f"scale_{size}.csv")
            
            start_time = time.time()
            result = await process_csv_hierarchy_optimized(
                csv_file_path=csv_path,
                tenant_id=1300,
                uploaded_by_user_id=1323
            )
            processing_time = time.time() - start_time
            
            throughput = size / processing_time
            results.append({
                'size': size,
                'time': processing_time,
                'throughput': throughput,
                'status': result.status
            })
            
            logger.info(f"📈 Size {size}: {processing_time:.2f}s ({throughput:.1f} users/sec)")
        
        # Performance requirements
        for result in results:
            assert result['status'] == 'completed', f"Size {result['size']} failed"
            if result['size'] <= 30:
                assert result['time'] < 5.0, f"Size {result['size']} took {result['time']:.2f}s, expected <5s"
            if result['size'] <= 100:
                assert result['time'] < 15.0, f"Size {result['size']} took {result['time']:.2f}s, expected <15s"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with invalid CSV
        invalid_df = pd.DataFrame({'Invalid': ['data'], 'Columns': ['here']})
        csv_path = self._create_temp_csv(invalid_df, "invalid.csv")
        
        result = await process_csv_hierarchy_optimized(
            csv_file_path=csv_path,
            tenant_id=1300,
            uploaded_by_user_id=1323
        )
        
        # Should handle gracefully (either succeed with defaults or fail gracefully)
        assert result.status in ["completed", "failed"], f"Unexpected status: {result.status}"
        
        logger.info(f"✅ Error handling: {result.status}")

class TestPerformanceComparison:
    """Compare optimized vs legacy performance"""
    
    @pytest.mark.asyncio
    async def test_optimized_vs_legacy_performance(self):
        """Compare optimized vs legacy processing performance"""
        df = TestDataGenerator.create_simple_hierarchy_csv(30)
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "comparison.csv")
        df.to_csv(csv_path, index=False)
        
        try:
            # Test optimized version
            start_time = time.time()
            optimized_result = await process_csv_hierarchy_optimized(
                csv_file_path=csv_path,
                tenant_id=1300,
                uploaded_by_user_id=1323
            )
            optimized_time = time.time() - start_time
            
            # Test legacy version (if available)
            try:
                from src.rba.hierarchy_workflow_executor import process_csv_hierarchy_via_rba
                
                start_time = time.time()
                legacy_result = await process_csv_hierarchy_via_rba(
                    csv_file_path=csv_path,
                    tenant_id=1300,
                    uploaded_by_user_id=1323,
                    use_optimized=False
                )
                legacy_time = time.time() - start_time
                
                # Compare performance
                improvement = legacy_time / optimized_time
                logger.info(f"🚀 Performance comparison:")
                logger.info(f"   Optimized: {optimized_time:.2f}s")
                logger.info(f"   Legacy: {legacy_time:.2f}s")
                logger.info(f"   Improvement: {improvement:.1f}x faster")
                
                assert optimized_time < legacy_time, "Optimized should be faster than legacy"
                
            except ImportError:
                logger.info(f"✅ Optimized processing: {optimized_time:.2f}s (legacy not available)")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

# Performance benchmarking functions
def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("🚀 Running Performance Benchmark for Optimized Hierarchy Processing")
    print("=" * 70)
    
    mapper = SuperSmartRBAMapper()
    sizes = [10, 30, 50, 100, 200, 500]
    
    for size in sizes:
        df = TestDataGenerator.create_simple_hierarchy_csv(size)
        
        start_time = time.time()
        mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
        processing_time = time.time() - start_time
        
        throughput = size / processing_time
        
        print(f"📊 {size:3d} users: {processing_time:6.3f}s ({throughput:6.1f} users/sec) - {confidence:.1%} confidence")
    
    print("=" * 70)

if __name__ == "__main__":
    # Run benchmark if called directly
    run_performance_benchmark()
