#!/usr/bin/env python3
"""
Quick Test Runner for Optimized Hierarchy Processing
===================================================
Run this script to quickly test the performance improvements.

Usage:
    python test_hierarchy_performance.py
"""

import asyncio
import pandas as pd
import time
import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_csv(num_users: int = 30) -> pd.DataFrame:
    """Create a test CSV with hierarchical data"""
    data = []
    
    # CEO
    data.append({
        'Name': 'John Smith',
        'Email': 'john.smith@company.com',
        'Job Title': 'Chief Executive Officer',
        'Manager Email': '',
        'Department': 'Executive',
        'Location': 'New York, NY'
    })
    
    # VPs
    vp_emails = []
    for i in range(3):
        email = f'vp{i+1}@company.com'
        vp_emails.append(email)
        data.append({
            'Name': f'VP {["Sales", "Engineering", "Marketing"][i]}',
            'Email': email,
            'Job Title': 'Vice President',
            'Manager Email': 'john.smith@company.com',
            'Department': ['Sales', 'Engineering', 'Marketing'][i],
            'Location': ['San Francisco, CA', 'Austin, TX', 'Boston, MA'][i]
        })
    
    # Directors
    director_emails = []
    for i in range(6):
        email = f'director{i+1}@company.com'
        director_emails.append(email)
        data.append({
            'Name': f'Director {i+1}',
            'Email': email,
            'Job Title': 'Director',
            'Manager Email': vp_emails[i % 3],
            'Department': ['Sales', 'Engineering', 'Marketing'][i % 3],
            'Location': ['San Francisco, CA', 'Austin, TX', 'Boston, MA'][i % 3]
        })
    
    # Fill remaining with employees
    remaining = num_users - len(data)
    for i in range(remaining):
        manager_email = director_emails[i % len(director_emails)] if director_emails else vp_emails[i % len(vp_emails)]
        data.append({
            'Name': f'Employee {i+1}',
            'Email': f'employee{i+1}@company.com',
            'Job Title': 'Individual Contributor',
            'Manager Email': manager_email,
            'Department': ['Sales', 'Engineering', 'Marketing'][i % 3],
            'Location': ['San Francisco, CA', 'Austin, TX', 'Boston, MA'][i % 3]
        })
    
    return pd.DataFrame(data)

def test_super_smart_mapper():
    """Test the Super Smart RBA Mapper"""
    print("🧠 Testing Super Smart RBA Mapper...")
    
    try:
        from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
        
        mapper = SuperSmartRBAMapper()
        df = create_test_csv(30)
        
        print(f"   📊 Input: {len(df)} users with columns: {list(df.columns)}")
        
        start_time = time.time()
        mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Processed in {processing_time:.3f}s")
        print(f"   🎯 Confidence: {confidence:.1%}")
        print(f"   🔍 Detected system: {detected_system}")
        print(f"   📈 Throughput: {len(df)/processing_time:.1f} users/sec")
        
        # Check output quality
        required_fields = ['Name', 'Email', 'Region', 'Level']
        missing_fields = [field for field in required_fields if field not in mapped_df.columns]
        if missing_fields:
            print(f"   ⚠️  Missing fields: {missing_fields}")
        else:
            print(f"   ✅ All required fields present")
        
        return processing_time, confidence
        
    except ImportError as e:
        print(f"   ❌ Super Smart Mapper not available: {e}")
        return None, None

def test_optimized_mapper():
    """Test the Optimized Universal Mapper"""
    print("\n⚡ Testing Optimized Universal Mapper...")
    
    try:
        from hierarchy_processor.core.optimized_universal_mapper import OptimizedUniversalMapper
        
        mapper = OptimizedUniversalMapper(enable_caching=True, chunk_size=100)
        df = create_test_csv(50)
        
        print(f"   📊 Input: {len(df)} users")
        
        start_time = time.time()
        mapped_df = mapper.map_any_hrms_to_crenovent_vectorized(df)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Vectorized processing in {processing_time:.3f}s")
        print(f"   📈 Throughput: {len(df)/processing_time:.1f} users/sec")
        print(f"   📋 Output columns: {len(mapped_df.columns)}")
        
        return processing_time
        
    except ImportError as e:
        print(f"   ❌ Optimized Mapper not available: {e}")
        return None

async def test_complete_workflow():
    """Test the complete optimized workflow"""
    print("\n🚀 Testing Complete Optimized Workflow...")
    
    try:
        from src.rba.hierarchy_workflow_executor import process_csv_hierarchy_optimized
        
        # Create test CSV file
        df = create_test_csv(30)
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "test_30_users.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"   📁 Created test CSV: {csv_path}")
        print(f"   👥 Users: {len(df)}")
        
        start_time = time.time()
        
        result = await process_csv_hierarchy_optimized(
            csv_file_path=csv_path,
            tenant_id=1300,
            uploaded_by_user_id=1323
        )
        
        total_time = time.time() - start_time
        
        print(f"   ⏱️  Total processing time: {total_time:.2f}s")
        print(f"   📊 Status: {result.status}")
        
        if result.status == "completed":
            print(f"   ✅ SUCCESS: 30 users processed in {total_time:.2f}s")
            if total_time < 5.0:
                print(f"   🎯 PERFORMANCE TARGET MET: <5 seconds ✅")
            else:
                print(f"   ⚠️  Performance target missed: {total_time:.2f}s > 5s")
        else:
            print(f"   ❌ FAILED: {result.error}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return total_time, result.status
        
    except ImportError as e:
        print(f"   ❌ Complete workflow not available: {e}")
        return None, None

def run_performance_benchmark():
    """Run performance benchmark with different dataset sizes"""
    print("\n📈 Performance Benchmark...")
    
    try:
        from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
        
        mapper = SuperSmartRBAMapper()
        sizes = [10, 30, 50, 100, 200]
        
        print("   Size | Time (s) | Throughput (users/sec) | Confidence")
        print("   -----|----------|------------------------|------------")
        
        for size in sizes:
            df = create_test_csv(size)
            
            start_time = time.time()
            mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
            processing_time = time.time() - start_time
            
            throughput = size / processing_time
            
            print(f"   {size:4d} | {processing_time:8.3f} | {throughput:20.1f} | {confidence:10.1%}")
        
    except ImportError as e:
        print(f"   ❌ Benchmark not available: {e}")

def main():
    """Main test runner"""
    print("🧪 Optimized Hierarchy Processing Test Suite")
    print("=" * 50)
    
    # Test individual components
    smart_time, smart_confidence = test_super_smart_mapper()
    optimized_time = test_optimized_mapper()
    
    # Test complete workflow
    workflow_time, workflow_status = asyncio.run(test_complete_workflow())
    
    # Run benchmark
    run_performance_benchmark()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    if smart_time and smart_confidence:
        print(f"✅ Super Smart Mapper: {smart_time:.3f}s, {smart_confidence:.1%} confidence")
    
    if optimized_time:
        print(f"✅ Optimized Mapper: {optimized_time:.3f}s")
    
    if workflow_time and workflow_status:
        print(f"✅ Complete Workflow: {workflow_time:.2f}s, {workflow_status}")
        
        if workflow_status == "completed" and workflow_time < 5.0:
            print("\n🎉 PERFORMANCE TARGET ACHIEVED!")
            print(f"   30 users processed in {workflow_time:.2f}s (target: <5s)")
        elif workflow_status == "completed":
            print(f"\n⚠️  Performance target missed: {workflow_time:.2f}s > 5s")
        else:
            print(f"\n❌ Workflow failed: {workflow_status}")
    
    print("\n🏁 Testing complete!")

if __name__ == "__main__":
    main()
