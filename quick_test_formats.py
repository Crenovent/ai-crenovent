#!/usr/bin/env python3
"""
Quick Test Runner for Dynamic CSV Formats
========================================
Simple script to quickly test both CSV formats with basic logging.
"""

import pandas as pd
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_format(csv_path: str, format_name: str):
    """Test a specific CSV format"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {format_name}")
    print(f"📁 File: {csv_path}")
    print('='*60)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Test Super Smart Mapper
    try:
        from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
        
        print(f"\n🧠 Testing Super Smart RBA Mapper...")
        mapper = SuperSmartRBAMapper()
        
        start_time = time.time()
        mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
        processing_time = time.time() - start_time
        
        throughput = len(df) / processing_time
        
        print(f"✅ Processing Results:")
        print(f"   ⏱️  Time: {processing_time:.3f}s")
        print(f"   🎯 Confidence: {confidence:.1%}")
        print(f"   🔍 System: {detected_system}")
        print(f"   📈 Throughput: {throughput:.1f} records/sec")
        print(f"   📊 Output: {len(mapped_df)} records, {len(mapped_df.columns)} columns")
        
        # Check key fields
        key_fields = ['Name', 'Email', 'Reporting Email', 'Role Title', 'Region']
        present_fields = [field for field in key_fields if field in mapped_df.columns]
        missing_fields = [field for field in key_fields if field not in mapped_df.columns]
        
        print(f"   ✅ Present: {present_fields}")
        if missing_fields:
            print(f"   ⚠️  Missing: {missing_fields}")
        
        # Show sample mapping
        print(f"\n📋 Sample Mapping (first 2 records):")
        for i in range(min(2, len(mapped_df))):
            sample = {}
            for field in ['Name', 'Email', 'Reporting Email', 'Role Title']:
                if field in mapped_df.columns:
                    sample[field] = mapped_df.iloc[i][field]
            print(f"   Record {i+1}: {sample}")
        
        return {
            'success': True,
            'time': processing_time,
            'confidence': confidence,
            'system': detected_system,
            'throughput': throughput
        }
        
    except ImportError:
        print(f"❌ Super Smart RBA Mapper not available")
        return {'success': False, 'error': 'Import failed'}
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main test runner"""
    print("🚀 Quick Dynamic CSV Format Test")
    print("="*60)
    
    # Test both formats
    test_files = [
        ('test_data/innovatetech_sales_hierarchy.csv', 'InnovateTech Sales (Contact Name, Email ID, Reporting Manager)'),
        ('test_data/acmecorp_employee_hierarchy.csv', 'AcmeCorp Employee (Full Name, Email, Manager Email)')
    ]
    
    results = []
    
    for csv_path, format_name in test_files:
        result = test_format(csv_path, format_name)
        if result:
            results.append({
                'format': format_name,
                'path': csv_path,
                **result
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print('='*60)
    
    successful_tests = [r for r in results if r['success']]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    
    if successful_tests:
        avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
        avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests)
        avg_throughput = sum(r['throughput'] for r in successful_tests) / len(successful_tests)
        
        print(f"⏱️  Average time: {avg_time:.3f}s")
        print(f"🎯 Average confidence: {avg_confidence:.1%}")
        print(f"📈 Average throughput: {avg_throughput:.1f} records/sec")
        
        # Performance check
        max_time = max(r['time'] for r in successful_tests)
        if max_time < 1.0:
            print(f"🎯 EXCELLENT: All formats process in <1 second ✅")
        elif max_time < 5.0:
            print(f"✅ GOOD: All formats meet 5-second target")
        else:
            print(f"⚠️  Some formats exceed 5-second target")
        
        # Confidence check
        min_confidence = min(r['confidence'] for r in successful_tests)
        if min_confidence > 0.9:
            print(f"🎯 EXCELLENT: All formats have >90% confidence ✅")
        elif min_confidence > 0.8:
            print(f"✅ GOOD: All formats have >80% confidence")
        else:
            print(f"⚠️  Some formats have low confidence")
    
    print(f"\n🏁 Testing complete!")
    print(f"💡 Run 'python test_dynamic_csv_formats.py' for detailed logging")

if __name__ == "__main__":
    main()
