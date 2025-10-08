#!/usr/bin/env python3
"""
Test Reporting Email Logic for Hierarchy Building

This script specifically tests the reporting email relationship processing
to ensure hierarchy is built correctly based on manager-employee relationships.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from hierarchy_processor.core.improved_hierarchy_builder import ImprovedHierarchyBuilder
from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
from hierarchy_processor.core.optimized_universal_mapper import OptimizedUniversalMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reporting_email_test.log')
    ]
)
logger = logging.getLogger(__name__)

def create_test_hierarchy_data():
    """Create test data with clear reporting relationships"""
    test_data = [
        # CEO - No manager
        {"Name": "Sarah Chen", "Email": "sarah.chen@testcorp.com", "Role Title": "CEO", "Manager Email": ""},
        
        # VPs report to CEO
        {"Name": "Michael Rodriguez", "Email": "michael.rodriguez@testcorp.com", "Role Title": "VP Sales", "Manager Email": "sarah.chen@testcorp.com"},
        {"Name": "Jennifer Wang", "Email": "jennifer.wang@testcorp.com", "Role Title": "VP Marketing", "Manager Email": "sarah.chen@testcorp.com"},
        {"Name": "David Kim", "Email": "david.kim@testcorp.com", "Role Title": "VP Engineering", "Manager Email": "sarah.chen@testcorp.com"},
        
        # Directors report to VPs
        {"Name": "Lisa Thompson", "Email": "lisa.thompson@testcorp.com", "Role Title": "Sales Director", "Manager Email": "michael.rodriguez@testcorp.com"},
        {"Name": "Robert Brown", "Email": "robert.brown@testcorp.com", "Role Title": "Marketing Director", "Manager Email": "jennifer.wang@testcorp.com"},
        {"Name": "Amanda Davis", "Email": "amanda.davis@testcorp.com", "Role Title": "Engineering Director", "Manager Email": "david.kim@testcorp.com"},
        
        # Managers report to Directors
        {"Name": "John Smith", "Email": "john.smith@testcorp.com", "Role Title": "Sales Manager", "Manager Email": "lisa.thompson@testcorp.com"},
        {"Name": "Emily Johnson", "Email": "emily.johnson@testcorp.com", "Role Title": "Marketing Manager", "Manager Email": "robert.brown@testcorp.com"},
        {"Name": "Chris Wilson", "Email": "chris.wilson@testcorp.com", "Role Title": "Engineering Manager", "Manager Email": "amanda.davis@testcorp.com"},
        
        # Individual Contributors report to Managers
        {"Name": "Alex Garcia", "Email": "alex.garcia@testcorp.com", "Role Title": "Account Executive", "Manager Email": "john.smith@testcorp.com"},
        {"Name": "Maria Lopez", "Email": "maria.lopez@testcorp.com", "Role Title": "Marketing Specialist", "Manager Email": "emily.johnson@testcorp.com"},
        {"Name": "James Taylor", "Email": "james.taylor@testcorp.com", "Role Title": "Software Engineer", "Manager Email": "chris.wilson@testcorp.com"},
        
        # Test edge cases
        {"Name": "Orphan Employee", "Email": "orphan@testcorp.com", "Role Title": "Consultant", "Manager Email": "nonexistent@testcorp.com"},  # Manager not in dataset
        {"Name": "No Manager", "Email": "nomanager@testcorp.com", "Role Title": "Contractor", "Manager Email": ""},  # No manager specified
    ]
    
    return pd.DataFrame(test_data)

def test_field_mapping_for_reporting_email():
    """Test that reporting email fields are correctly mapped"""
    logger.info("=" * 80)
    logger.info("TESTING FIELD MAPPING FOR REPORTING EMAIL")
    logger.info("=" * 80)
    
    # Create test CSV with different column name variations
    test_variations = [
        # Variation 1: Standard naming
        {"Name": "John Doe", "Email": "john@test.com", "Manager Email": "jane@test.com", "Role Title": "Manager"},
        
        # Variation 2: Different naming conventions
        {"Full Name": "Jane Smith", "Email Address": "jane@test.com", "Reporting Manager": "boss@test.com", "Job Title": "Director"},
        
        # Variation 3: Salesforce-style naming
        {"Contact Name": "Bob Wilson", "Email ID": "bob@test.com", "Reports To Email": "supervisor@test.com", "Designation": "VP"},
        
        # Variation 4: BambooHR-style naming
        {"Employee Name": "Alice Brown", "Work Email": "alice@test.com", "Supervisor Email": "chief@test.com", "Position": "Engineer"},
    ]
    
    for i, variation in enumerate(test_variations, 1):
        logger.info(f"\n--- Testing Variation {i}: {list(variation.keys())} ---")
        
        df = pd.DataFrame([variation])
        logger.info(f"Input columns: {list(df.columns)}")
        
        # Test SuperSmartRBAMapper
        mapper = SuperSmartRBAMapper()
        result_df, confidence, system = mapper.map_csv_intelligently(df)
        
        logger.info(f"Mapped columns: {list(result_df.columns)}")
        logger.info(f"Confidence: {confidence:.1f}%")
        logger.info(f"Detected system: {system}")
        
        # Check if reporting email was mapped correctly
        if 'Reporting Email' in result_df.columns:
            reporting_email = result_df['Reporting Email'].iloc[0]
            logger.info(f"✅ Reporting Email mapped: '{reporting_email}'")
        else:
            logger.error("❌ Reporting Email not mapped!")
        
        # Check other key fields
        for field in ['Name', 'Email', 'Role Title']:
            if field in result_df.columns:
                value = result_df[field].iloc[0]
                logger.info(f"✅ {field} mapped: '{value}'")
            else:
                logger.error(f"❌ {field} not mapped!")

def test_hierarchy_building_logic():
    """Test the actual hierarchy building from reporting email relationships"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING HIERARCHY BUILDING LOGIC")
    logger.info("=" * 80)
    
    # Create test data
    df = create_test_hierarchy_data()
    logger.info(f"Created test data with {len(df)} employees")
    
    # First, map the fields using SuperSmartRBAMapper
    mapper = SuperSmartRBAMapper()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    
    logger.info(f"Field mapping confidence: {confidence:.1f}%")
    logger.info(f"Mapped columns: {list(mapped_df.columns)}")
    
    # Now test hierarchy building
    hierarchy_builder = ImprovedHierarchyBuilder()
    
    try:
        root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
        
        logger.info(f"\n📊 HIERARCHY BUILDING RESULTS:")
        logger.info(f"Root nodes found: {len(root_nodes)}")
        logger.info(f"Valid relationships: {len(validation_result.valid_relationships)}")
        logger.info(f"Invalid emails: {len(validation_result.invalid_emails)}")
        logger.info(f"Missing managers: {len(validation_result.missing_managers)}")
        logger.info(f"Circular references: {len(validation_result.circular_references)}")
        logger.info(f"Hierarchy health score: {validation_result.hierarchy_health_score:.2f}")
        
        # Log root nodes
        logger.info(f"\n🏆 ROOT NODES (Org Leaders):")
        for root in root_nodes:
            logger.info(f"  - {root.name} ({root.email}) - {root.title}")
        
        # Log reporting relationships
        logger.info(f"\n📋 VALID REPORTING RELATIONSHIPS:")
        for rel in validation_result.valid_relationships[:10]:  # Show first 10
            logger.info(f"  - {rel['employee']} reports to {rel['manager']}")
        
        if len(validation_result.valid_relationships) > 10:
            logger.info(f"  ... and {len(validation_result.valid_relationships) - 10} more")
        
        # Log missing managers
        if validation_result.missing_managers:
            logger.info(f"\n⚠️ MISSING MANAGERS:")
            for missing in validation_result.missing_managers:
                logger.info(f"  - {missing['employee']} -> {missing['manager']} (not found)")
        
        # Log circular references
        if validation_result.circular_references:
            logger.error(f"\n🔄 CIRCULAR REFERENCES DETECTED:")
            for cycle in validation_result.circular_references:
                logger.error(f"  - Cycle: {' -> '.join(cycle)}")
        
        # Test hierarchy traversal
        logger.info(f"\n🌳 HIERARCHY STRUCTURE:")
        for root in root_nodes:
            _print_hierarchy_tree(root, level=0)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hierarchy building failed: {e}")
        return False

def _print_hierarchy_tree(node, level=0):
    """Recursively print hierarchy tree structure"""
    indent = "  " * level
    logger.info(f"{indent}├─ {node.name} ({node.title}) - Level: {node.level}")
    
    for child in node.children:
        _print_hierarchy_tree(child, level + 1)

def test_edge_cases():
    """Test edge cases in reporting email logic"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING EDGE CASES")
    logger.info("=" * 80)
    
    edge_cases = [
        # Case 1: Circular reference
        {
            "data": [
                {"Name": "Alice", "Email": "alice@test.com", "Manager Email": "bob@test.com", "Role Title": "Manager"},
                {"Name": "Bob", "Email": "bob@test.com", "Manager Email": "charlie@test.com", "Role Title": "Director"},
                {"Name": "Charlie", "Email": "charlie@test.com", "Manager Email": "alice@test.com", "Role Title": "VP"},
            ],
            "description": "Circular reference: Alice -> Bob -> Charlie -> Alice"
        },
        
        # Case 2: Self-reporting
        {
            "data": [
                {"Name": "Self Reporter", "Email": "self@test.com", "Manager Email": "self@test.com", "Role Title": "Manager"},
            ],
            "description": "Self-reporting employee"
        },
        
        # Case 3: Multiple root nodes
        {
            "data": [
                {"Name": "CEO 1", "Email": "ceo1@test.com", "Manager Email": "", "Role Title": "CEO"},
                {"Name": "CEO 2", "Email": "ceo2@test.com", "Manager Email": "", "Role Title": "CEO"},
                {"Name": "Employee", "Email": "emp@test.com", "Manager Email": "ceo1@test.com", "Role Title": "Manager"},
            ],
            "description": "Multiple root nodes (CEOs)"
        },
        
        # Case 4: Invalid email formats
        {
            "data": [
                {"Name": "Valid", "Email": "valid@test.com", "Manager Email": "invalid-email", "Role Title": "Manager"},
                {"Name": "Invalid", "Email": "not-an-email", "Manager Email": "valid@test.com", "Role Title": "Employee"},
            ],
            "description": "Invalid email formats"
        }
    ]
    
    for i, case in enumerate(edge_cases, 1):
        logger.info(f"\n--- Edge Case {i}: {case['description']} ---")
        
        df = pd.DataFrame(case['data'])
        
        try:
            # Map fields
            mapper = SuperSmartRBAMapper()
            mapped_df, confidence, system = mapper.map_csv_intelligently(df)
            
            # Build hierarchy
            hierarchy_builder = ImprovedHierarchyBuilder()
            root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
            
            logger.info(f"✅ Processed successfully:")
            logger.info(f"  - Root nodes: {len(root_nodes)}")
            logger.info(f"  - Valid relationships: {len(validation_result.valid_relationships)}")
            logger.info(f"  - Circular references: {len(validation_result.circular_references)}")
            logger.info(f"  - Health score: {validation_result.hierarchy_health_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Edge case failed: {e}")

def test_performance_with_large_dataset():
    """Test performance with a larger dataset"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PERFORMANCE WITH LARGE DATASET")
    logger.info("=" * 80)
    
    import time
    
    # Generate larger test dataset
    large_data = []
    
    # Create a 5-level hierarchy with 100 employees
    for i in range(100):
        if i == 0:
            # CEO
            employee = {
                "Name": f"CEO {i}",
                "Email": f"ceo{i}@testcorp.com",
                "Manager Email": "",
                "Role Title": "CEO"
            }
        elif i <= 5:
            # VPs report to CEO
            employee = {
                "Name": f"VP {i}",
                "Email": f"vp{i}@testcorp.com",
                "Manager Email": "ceo0@testcorp.com",
                "Role Title": "VP"
            }
        elif i <= 20:
            # Directors report to VPs
            vp_index = ((i - 6) % 5) + 1
            employee = {
                "Name": f"Director {i}",
                "Email": f"director{i}@testcorp.com",
                "Manager Email": f"vp{vp_index}@testcorp.com",
                "Role Title": "Director"
            }
        elif i <= 50:
            # Managers report to Directors
            director_index = ((i - 21) % 15) + 6
            employee = {
                "Name": f"Manager {i}",
                "Email": f"manager{i}@testcorp.com",
                "Manager Email": f"director{director_index}@testcorp.com",
                "Role Title": "Manager"
            }
        else:
            # ICs report to Managers
            manager_index = ((i - 51) % 30) + 21
            employee = {
                "Name": f"Employee {i}",
                "Email": f"employee{i}@testcorp.com",
                "Manager Email": f"manager{manager_index}@testcorp.com",
                "Role Title": "Individual Contributor"
            }
        
        large_data.append(employee)
    
    df = pd.DataFrame(large_data)
    logger.info(f"Generated dataset with {len(df)} employees")
    
    # Test field mapping performance
    start_time = time.time()
    mapper = SuperSmartRBAMapper()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    mapping_time = time.time() - start_time
    
    logger.info(f"Field mapping completed in {mapping_time:.3f} seconds")
    logger.info(f"Throughput: {len(df) / mapping_time:.1f} records/second")
    
    # Test hierarchy building performance
    start_time = time.time()
    hierarchy_builder = ImprovedHierarchyBuilder()
    root_nodes, validation_result = hierarchy_builder.build_hierarchy_from_dataframe(mapped_df)
    hierarchy_time = time.time() - start_time
    
    logger.info(f"Hierarchy building completed in {hierarchy_time:.3f} seconds")
    logger.info(f"Total processing time: {mapping_time + hierarchy_time:.3f} seconds")
    logger.info(f"Overall throughput: {len(df) / (mapping_time + hierarchy_time):.1f} records/second")
    
    # Validate results
    logger.info(f"\n📊 LARGE DATASET RESULTS:")
    logger.info(f"Root nodes: {len(root_nodes)}")
    logger.info(f"Valid relationships: {len(validation_result.valid_relationships)}")
    logger.info(f"Health score: {validation_result.hierarchy_health_score:.2f}")

def main():
    """Run all reporting email logic tests"""
    logger.info("🧪 REPORTING EMAIL LOGIC TEST SUITE")
    logger.info("=" * 80)
    
    try:
        # Test 1: Field mapping
        test_field_mapping_for_reporting_email()
        
        # Test 2: Hierarchy building
        success = test_hierarchy_building_logic()
        
        # Test 3: Edge cases
        test_edge_cases()
        
        # Test 4: Performance
        test_performance_with_large_dataset()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL REPORTING EMAIL TESTS COMPLETED")
        logger.info("=" * 80)
        
        if success:
            logger.info("🎉 Reporting email logic is working correctly!")
        else:
            logger.error("❌ Some tests failed - check logs for details")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main()
