#!/usr/bin/env python3
"""
Test script to verify the RBA → LLM fallback flow works correctly
"""

import pandas as pd
import requests
import json
import sys
import os

def test_rba_success():
    """Test with a CSV that should work with RBA Universal Mapper"""
    print("🧪 Test 1: RBA Universal Mapper Success")
    
    # Standard Workday format that should work with RBA
    test_data = [
        {
            "first_name": "John",
            "last_name": "Smith", 
            "email": "john.smith@company.com",
            "business_title": "Senior Account Executive",
            "managername": "Jane Doe",
            "division": "Sales"
        },
        {
            "first_name": "Jane",
            "last_name": "Doe",
            "email": "jane.doe@company.com", 
            "business_title": "Sales Director",
            "managername": "Robert Brown",
            "division": "Sales"
        }
    ]
    
    return test_csv_processing(test_data, "RBA Universal Mapper")

def test_llm_fallback():
    """Test with a CSV that should trigger LLM fallback"""
    print("\n🧪 Test 2: LLM Fallback Trigger")
    
    # Unusual column names that might challenge RBA
    test_data = [
        {
            "PersonFullName": "Mike Johnson",
            "WorkEmailAddress": "mike@company.com",
            "JobRole": "Account Manager", 
            "DirectSupervisor": "Jane Doe",
            "OrganizationalUnit": "Revenue Team"
        },
        {
            "PersonFullName": "Sarah Wilson", 
            "WorkEmailAddress": "sarah@company.com",
            "JobRole": "Customer Success Manager",
            "DirectSupervisor": "Mike Johnson", 
            "OrganizationalUnit": "Success Team"
        }
    ]
    
    return test_csv_processing(test_data, "LLM Fallback")

def test_csv_processing(csv_data, expected_processor):
    """Test CSV processing with the universal endpoint"""
    try:
        # Prepare request
        request_data = {
            "csv_data": csv_data,
            "tenant_id": 1300,
            "uploaded_by": 1323
        }
        
        # Send to universal endpoint
        response = requests.post(
            "http://localhost:8000/api/hierarchy/normalize-csv-universal",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Success! Processed by: {result.get('detected_system', 'unknown')}")
            print(f"📊 Confidence: {result.get('confidence', 0)}")
            print(f"📝 Records processed: {len(result.get('normalized_data', []))}")
            
            # Check if we got valid data
            normalized_data = result.get('normalized_data', [])
            if normalized_data:
                sample_record = normalized_data[0]
                print(f"👤 Sample Name: {sample_record.get('Name', 'Missing')}")
                print(f"📧 Sample Email: {sample_record.get('Email', 'Missing')}")
                print(f"💼 Sample Role: {sample_record.get('Role Title', 'Missing')}")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run all fallback tests"""
    print("🚀 Testing RBA → LLM Fallback Flow")
    print("=" * 50)
    
    # Check if Python service is running
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code != 200:
            print("❌ Python service not running on localhost:8000")
            print("Please start the service with: python main.py")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Python service on localhost:8000")
        print("Please start the service with: python main.py")
        sys.exit(1)
    
    print("✅ Python service is running")
    
    # Run tests
    test1_passed = test_rba_success()
    test2_passed = test_llm_fallback()
    
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary:")
    print(f"Test 1 (RBA Success): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (LLM Fallback): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! RBA → LLM fallback flow is working correctly!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
