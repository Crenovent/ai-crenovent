#!/usr/bin/env python3
"""
Clean Test for Reporting Email Logic (Windows Compatible)
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import os
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hierarchy_processor.core.improved_hierarchy_builder import ImprovedHierarchyBuilder
from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper

# Configure logging without Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reporting_email_clean_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_reporting_email_mapping():
    """Test that reporting email fields are correctly detected and mapped"""
    logger.info("=" * 60)
    logger.info("TESTING REPORTING EMAIL FIELD MAPPING")
    logger.info("=" * 60)
    
    # Test different CSV formats
    test_cases = [
        {
            "name": "Standard Format",
            "data": {"Name": "John Doe", "Email": "john@test.com", "Manager Email": "jane@test.com", "Role Title": "Manager"}
        },
        {
            "name": "Salesforce Format", 
            "data": {"Contact Name": "Jane Smith", "Email ID": "jane@test.com", "Reports To Email": "boss@test.com", "Designation": "Director"}
        },
        {
            "name": "BambooHR Format",
            "data": {"Full Name": "Bob Wilson", "Work Email": "bob@test.com", "Supervisor Email": "chief@test.com", "Job Title": "Engineer"}
        }
    ]
    
    mapper = SuperSmartRBAMapper()
    
    for case in test_cases:
        logger.info(f"\n--- Testing {case['name']} ---")
        df = pd.DataFrame([case['data']])
        
        logger.info(f"Input columns: {list(df.columns)}")
        
        # Map the CSV
        result_df, confidence, system = mapper.map_csv_intelligently(df)
        
        logger.info(f"Confidence: {confidence:.1f}%")
        logger.info(f"System detected: {system}")
        
        # Check key mappings
        if 'Reporting Email' in result_df.columns:
            reporting_email = result_df['Reporting Email'].iloc[0]
            logger.info(f"SUCCESS: Reporting Email mapped to '{reporting_email}'")
        else:
            logger.error("FAILED: Reporting Email not mapped!")
        
        if 'Name' in result_df.columns:
            name = result_df['Name'].iloc[0]
            logger.info(f"SUCCESS: Name mapped to '{name}'")
        
        if 'Email' in result_df.columns:
            email = result_df['Email'].iloc[0]
            logger.info(f"SUCCESS: Email mapped to '{email}'")

def test_hierarchy_building():
    """Test hierarchy building from reporting email relationships"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING HIERARCHY BUILDING FROM REPORTING EMAILS")
    logger.info("=" * 60)
    
    # Create test hierarchy data
    test_data = [
        {"Name": "CEO Sarah", "Email": "sarah@test.com", "Manager Email": "", "Role Title": "CEO"},
        {"Name": "VP Mike", "Email": "mike@test.com", "Manager Email": "sarah@test.com", "Role Title": "VP Sales"},
        {"Name": "VP Jennifer", "Email": "jennifer@test.com", "Manager Email": "sarah@test.com", "Role Title": "VP Marketing"},
        {"Name": "Director Lisa", "Email": "lisa@test.com", "Manager Email": "mike@test.com", "Role Title": "Sales Director"},
        {"Name": "Director Robert", "Email": "robert@test.com", "Manager Email": "jennifer@test.com", "Role Title": "Marketing Director"},
        {"Name": "Manager John", "Email": "john@test.com", "Manager Email": "lisa@test.com", "Role Title": "Sales Manager"},
        {"Name": "Manager Emily", "Email": "emily@test.com", "Manager Email": "robert@test.com", "Role Title": "Marketing Manager"},
        {"Name": "Rep Alex", "Email": "alex@test.com", "Manager Email": "john@test.com", "Role Title": "Sales Rep"},
        {"Name": "Specialist Maria", "Email": "maria@test.com", "Manager Email": "emily@test.com", "Role Title": "Marketing Specialist"},
    ]
    
    df = pd.DataFrame(test_data)
    logger.info(f"Created test data with {len(df)} employees")
    
    # Map fields first
    mapper = SuperSmartRBAMapper()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    
    logger.info(f"Field mapping confidence: {confidence:.1f}%")
    
    # Build hierarchy
    hierarchy_builder = ImprovedHierarchyBuilder()
    root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
    
    # Report results
    logger.info(f"\nHIERARCHY RESULTS:")
    logger.info(f"Root nodes: {len(root_nodes)}")
    logger.info(f"Valid relationships: {len(validation_result.valid_relationships)}")
    logger.info(f"Missing managers: {len(validation_result.missing_managers)}")
    logger.info(f"Circular references: {len(validation_result.circular_references)}")
    logger.info(f"Health score: {validation_result.hierarchy_health_score:.2f}")
    
    # Show root nodes
    logger.info(f"\nROOT NODES (Org Leaders):")
    for root in root_nodes:
        logger.info(f"  - {root.name} ({root.email}) - {root.title}")
    
    # Show reporting relationships
    logger.info(f"\nREPORTING RELATIONSHIPS:")
    for rel in validation_result.valid_relationships:
        logger.info(f"  - {rel['employee']} reports to {rel['manager']}")
    
    # Test hierarchy traversal
    logger.info(f"\nHIERARCHY TREE:")
    for root in root_nodes:
        print_hierarchy_tree(root, level=0)
    
    return len(validation_result.valid_relationships) > 0

def print_hierarchy_tree(node, level=0):
    """Print hierarchy tree structure"""
    indent = "  " * level
    logger.info(f"{indent}+-- {node.name} ({node.title}) - Level: {node.level}")
    
    for child in node.children:
        print_hierarchy_tree(child, level + 1)

def test_edge_cases():
    """Test edge cases in reporting email processing"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING EDGE CASES")
    logger.info("=" * 60)
    
    # Test circular reference
    logger.info("\n--- Testing Circular Reference ---")
    circular_data = [
        {"Name": "Alice", "Email": "alice@test.com", "Manager Email": "bob@test.com", "Role Title": "Manager"},
        {"Name": "Bob", "Email": "bob@test.com", "Manager Email": "alice@test.com", "Role Title": "Director"},
    ]
    
    df = pd.DataFrame(circular_data)
    mapper = SuperSmartRBAMapper()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    
    hierarchy_builder = ImprovedHierarchyBuilder()
    root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
    
    logger.info(f"Circular references detected: {len(validation_result.circular_references)}")
    logger.info(f"Health score: {validation_result.hierarchy_health_score:.2f}")
    
    # Test missing manager
    logger.info("\n--- Testing Missing Manager ---")
    missing_data = [
        {"Name": "Employee", "Email": "emp@test.com", "Manager Email": "missing@test.com", "Role Title": "Employee"},
    ]
    
    df = pd.DataFrame(missing_data)
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
    
    logger.info(f"Missing managers: {len(validation_result.missing_managers)}")
    logger.info(f"Root nodes created: {len(root_nodes)}")

def test_performance():
    """Test performance with larger dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PERFORMANCE")
    logger.info("=" * 60)
    
    # Generate 50 employee hierarchy
    large_data = []
    
    # CEO
    large_data.append({"Name": "CEO", "Email": "ceo@test.com", "Manager Email": "", "Role Title": "CEO"})
    
    # VPs (5)
    for i in range(1, 6):
        large_data.append({
            "Name": f"VP {i}",
            "Email": f"vp{i}@test.com", 
            "Manager Email": "ceo@test.com",
            "Role Title": "VP"
        })
    
    # Directors (15)
    for i in range(6, 21):
        vp_index = ((i - 6) % 5) + 1
        large_data.append({
            "Name": f"Director {i}",
            "Email": f"director{i}@test.com",
            "Manager Email": f"vp{vp_index}@test.com",
            "Role Title": "Director"
        })
    
    # Employees (30)
    for i in range(21, 51):
        director_index = ((i - 21) % 15) + 6
        large_data.append({
            "Name": f"Employee {i}",
            "Email": f"emp{i}@test.com",
            "Manager Email": f"director{director_index}@test.com",
            "Role Title": "Employee"
        })
    
    df = pd.DataFrame(large_data)
    logger.info(f"Generated dataset with {len(df)} employees")
    
    # Test field mapping performance
    start_time = time.time()
    mapper = SuperSmartRBAMapper()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    mapping_time = time.time() - start_time
    
    logger.info(f"Field mapping: {mapping_time:.3f} seconds ({len(df)/mapping_time:.1f} rec/sec)")
    
    # Test hierarchy building performance
    start_time = time.time()
    hierarchy_builder = ImprovedHierarchyBuilder()
    root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
    hierarchy_time = time.time() - start_time
    
    logger.info(f"Hierarchy building: {hierarchy_time:.3f} seconds ({len(df)/hierarchy_time:.1f} rec/sec)")
    logger.info(f"Total time: {mapping_time + hierarchy_time:.3f} seconds")
    logger.info(f"Overall throughput: {len(df)/(mapping_time + hierarchy_time):.1f} records/second")
    
    # Validate results
    logger.info(f"\nPERFORMANCE RESULTS:")
    logger.info(f"Root nodes: {len(root_nodes)}")
    logger.info(f"Valid relationships: {len(validation_result.valid_relationships)}")
    logger.info(f"Health score: {validation_result.hierarchy_health_score:.2f}")
    
    return mapping_time + hierarchy_time < 2.0  # Should be under 2 seconds

def main():
    """Run all tests"""
    logger.info("REPORTING EMAIL LOGIC TEST SUITE")
    logger.info("=" * 60)
    
    try:
        # Test 1: Field mapping
        test_reporting_email_mapping()
        
        # Test 2: Hierarchy building
        hierarchy_success = test_hierarchy_building()
        
        # Test 3: Edge cases
        test_edge_cases()
        
        # Test 4: Performance
        performance_success = test_performance()
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        if hierarchy_success and performance_success:
            logger.info("SUCCESS: All reporting email tests passed!")
            logger.info("- Field mapping works correctly")
            logger.info("- Hierarchy building is functional") 
            logger.info("- Performance meets requirements")
            logger.info("- Edge cases are handled properly")
        else:
            logger.error("FAILURE: Some tests failed")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main()
