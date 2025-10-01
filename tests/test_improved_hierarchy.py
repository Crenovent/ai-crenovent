#!/usr/bin/env python3
"""
Test script for the improved hierarchy builder

This script tests the new hierarchy algorithm with various scenarios:
- Normal hierarchy
- Circular references
- Missing managers
- Orphaned employees
- Large datasets
"""

import sys
import os
import pandas as pd
import logging
from typing import List, Dict

# Add the hierarchy processor to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hierarchy_processor'))

from hierarchy_processor.core.improved_hierarchy_builder import ImprovedHierarchyBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_normal() -> pd.DataFrame:
    """Create normal hierarchy test data"""
    data = [
        {'Name': 'Alice CEO', 'Email': 'alice@company.com', 'Role Title': 'CEO', 'Reporting Email': ''},
        {'Name': 'Bob CTO', 'Email': 'bob@company.com', 'Role Title': 'CTO', 'Reporting Email': 'alice@company.com'},
        {'Name': 'Carol VP Eng', 'Email': 'carol@company.com', 'Role Title': 'VP Engineering', 'Reporting Email': 'bob@company.com'},
        {'Name': 'Dave Director', 'Email': 'dave@company.com', 'Role Title': 'Director', 'Reporting Email': 'carol@company.com'},
        {'Name': 'Eve Manager', 'Email': 'eve@company.com', 'Role Title': 'Manager', 'Reporting Email': 'dave@company.com'},
        {'Name': 'Frank Engineer', 'Email': 'frank@company.com', 'Role Title': 'Senior Engineer', 'Reporting Email': 'eve@company.com'},
        {'Name': 'Grace Engineer', 'Email': 'grace@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'eve@company.com'},
    ]
    return pd.DataFrame(data)

def create_test_data_circular() -> pd.DataFrame:
    """Create test data with circular references"""
    data = [
        {'Name': 'Alice', 'Email': 'alice@company.com', 'Role Title': 'Manager', 'Reporting Email': 'bob@company.com'},
        {'Name': 'Bob', 'Email': 'bob@company.com', 'Role Title': 'Manager', 'Reporting Email': 'charlie@company.com'},
        {'Name': 'Charlie', 'Email': 'charlie@company.com', 'Role Title': 'Manager', 'Reporting Email': 'alice@company.com'},
        {'Name': 'Dave', 'Email': 'dave@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'alice@company.com'},
    ]
    return pd.DataFrame(data)

def create_test_data_missing_managers() -> pd.DataFrame:
    """Create test data with missing managers"""
    data = [
        {'Name': 'Alice', 'Email': 'alice@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'missing@company.com'},
        {'Name': 'Bob', 'Email': 'bob@company.com', 'Role Title': 'Manager', 'Reporting Email': ''},
        {'Name': 'Charlie', 'Email': 'charlie@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'bob@company.com'},
        {'Name': 'Dave', 'Email': 'dave@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'external@other.com'},
    ]
    return pd.DataFrame(data)

def create_test_data_complex() -> pd.DataFrame:
    """Create complex test data with multiple issues"""
    data = [
        # Normal hierarchy
        {'Name': 'CEO Alice', 'Email': 'ceo@company.com', 'Role Title': 'CEO', 'Reporting Email': ''},
        {'Name': 'CTO Bob', 'Email': 'cto@company.com', 'Role Title': 'CTO', 'Reporting Email': 'ceo@company.com'},
        {'Name': 'CFO Carol', 'Email': 'cfo@company.com', 'Role Title': 'CFO', 'Reporting Email': 'ceo@company.com'},
        
        # Engineering team
        {'Name': 'VP Eng Dave', 'Email': 'vp-eng@company.com', 'Role Title': 'VP Engineering', 'Reporting Email': 'cto@company.com'},
        {'Name': 'Dir Eng Eve', 'Email': 'dir-eng@company.com', 'Role Title': 'Director Engineering', 'Reporting Email': 'vp-eng@company.com'},
        {'Name': 'Mgr Eng Frank', 'Email': 'mgr-eng@company.com', 'Role Title': 'Engineering Manager', 'Reporting Email': 'dir-eng@company.com'},
        {'Name': 'Sr Eng Grace', 'Email': 'sr-eng@company.com', 'Role Title': 'Senior Engineer', 'Reporting Email': 'mgr-eng@company.com'},
        {'Name': 'Eng Henry', 'Email': 'eng@company.com', 'Role Title': 'Engineer', 'Reporting Email': 'mgr-eng@company.com'},
        
        # Circular reference
        {'Name': 'Circular A', 'Email': 'circ-a@company.com', 'Role Title': 'Manager', 'Reporting Email': 'circ-b@company.com'},
        {'Name': 'Circular B', 'Email': 'circ-b@company.com', 'Role Title': 'Manager', 'Reporting Email': 'circ-a@company.com'},
        
        # Missing manager
        {'Name': 'Orphan', 'Email': 'orphan@company.com', 'Role Title': 'Analyst', 'Reporting Email': 'missing@company.com'},
        
        # No manager
        {'Name': 'Independent', 'Email': 'independent@company.com', 'Role Title': 'Consultant', 'Reporting Email': ''},
    ]
    return pd.DataFrame(data)

def test_hierarchy_builder(test_name: str, df: pd.DataFrame) -> None:
    """Test the hierarchy builder with given data"""
    logger.info(f"\n{'='*50}")
    logger.info(f"🧪 Testing: {test_name}")
    logger.info(f"{'='*50}")
    
    try:
        # Initialize hierarchy builder
        hierarchy_builder = ImprovedHierarchyBuilder()
        
        # Build hierarchy
        root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(df)
        
        # Generate report
        report = hierarchy_builder.generate_hierarchy_report()
        
        # Print results
        logger.info(f"📊 Test Results for {test_name}:")
        logger.info(f"   📈 Health Score: {validation_result.hierarchy_health_score:.2f}%")
        logger.info(f"   👥 Total Employees: {len(df)}")
        logger.info(f"   🌳 Root Nodes: {len(root_nodes)}")
        logger.info(f"   ✅ Valid Relationships: {len(validation_result.valid_relationships)}")
        logger.info(f"   🔄 Circular References: {len(validation_result.circular_references)}")
        logger.info(f"   ❓ Missing Managers: {len(validation_result.missing_managers)}")
        logger.info(f"   👤 Orphaned Employees: {len(validation_result.orphaned_employees)}")
        
        if validation_result.circular_references:
            logger.warning(f"   🔄 Circular References Detected:")
            for i, cycle in enumerate(validation_result.circular_references):
                logger.warning(f"      {i+1}. {' -> '.join(cycle)}")
        
        if validation_result.missing_managers:
            logger.warning(f"   ❓ Missing Managers:")
            for missing in validation_result.missing_managers[:5]:  # Show first 5
                logger.warning(f"      {missing['employee']} -> {missing['manager']}")
        
        # Print hierarchy structure
        logger.info(f"   🏗️ Hierarchy Structure:")
        for root in root_nodes:
            print_hierarchy_tree(root, indent=6)
        
        # Print recommendations
        if report["recommendations"]:
            logger.info(f"   💡 Recommendations:")
            for rec in report["recommendations"]:
                logger.info(f"      • {rec}")
        
        logger.info(f"✅ Test {test_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test {test_name} failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

def print_hierarchy_tree(node, indent=0):
    """Print hierarchy tree structure"""
    prefix = " " * indent
    level_info = f"[{node.level}]" if hasattr(node, 'level') and node.level else ""
    depth_info = f"(depth: {node.hierarchy_depth})" if hasattr(node, 'hierarchy_depth') else ""
    print(f"{prefix}├─ {node.name} {level_info} {depth_info}")
    
    for child in node.children:
        print_hierarchy_tree(child, indent + 3)

def run_performance_test():
    """Test with larger dataset"""
    logger.info(f"\n{'='*50}")
    logger.info(f"🚀 Performance Test: Large Dataset")
    logger.info(f"{'='*50}")
    
    # Create large dataset
    import random
    
    employees = []
    titles = ['CEO', 'CTO', 'CFO', 'VP', 'Director', 'Senior Manager', 'Manager', 'Team Lead', 'Senior Engineer', 'Engineer', 'Analyst']
    
    # Create CEO
    employees.append({
        'Name': f'CEO Person',
        'Email': f'ceo@company.com',
        'Role Title': 'CEO',
        'Reporting Email': ''
    })
    
    # Create hierarchy levels
    for level in range(1, 6):  # 5 levels deep
        for i in range(min(10, 2 ** level)):  # Exponential growth
            emp_id = len(employees)
            
            # Find a random manager from previous level
            if level == 1:
                manager_email = 'ceo@company.com'
            else:
                possible_managers = [emp for emp in employees if emp['Email'] != 'ceo@company.com']
                manager_email = random.choice(possible_managers)['Email'] if possible_managers else 'ceo@company.com'
            
            employees.append({
                'Name': f'Employee {emp_id}',
                'Email': f'emp{emp_id}@company.com',
                'Role Title': random.choice(titles),
                'Reporting Email': manager_email
            })
    
    df_large = pd.DataFrame(employees)
    logger.info(f"Created large dataset with {len(df_large)} employees")
    
    # Time the operation
    import time
    start_time = time.time()
    
    test_hierarchy_builder("Large Dataset", df_large)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
    logger.info(f"📊 Processing rate: {len(df_large)/processing_time:.1f} employees/second")

def main():
    """Run all hierarchy tests"""
    logger.info("🚀 Starting Improved Hierarchy Builder Tests")
    
    # Test 1: Normal hierarchy
    df_normal = create_test_data_normal()
    test_hierarchy_builder("Normal Hierarchy", df_normal)
    
    # Test 2: Circular references
    df_circular = create_test_data_circular()
    test_hierarchy_builder("Circular References", df_circular)
    
    # Test 3: Missing managers
    df_missing = create_test_data_missing_managers()
    test_hierarchy_builder("Missing Managers", df_missing)
    
    # Test 4: Complex scenario
    df_complex = create_test_data_complex()
    test_hierarchy_builder("Complex Scenario", df_complex)
    
    # Test 5: Performance test
    run_performance_test()
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 All tests completed!")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
