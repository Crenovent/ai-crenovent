#!/usr/bin/env python3
import requests
import json

# Test data similar to what frontend sends
test_data = [
    {
        "Name": "Test User",
        "Email": "test@example.com",
        "Modules": "Calendar,Lets Meet,Cruxx,RevAIPro Action Center,Rhythm of Business"
    }
]

# Prepare form data
form_data = {
    'processed_users': json.dumps(test_data),
    'tenant_id': '1342'
}

print("🧪 Testing Complete Setup endpoint...")
print(f"📤 Sending data: {form_data}")

try:
    response = requests.post(
        'http://localhost:8000/api/onboarding/complete-setup',
        data=form_data
    )
    
    print(f"📥 Response status: {response.status_code}")
    print(f"📥 Response body: {response.text}")
    
except Exception as e:
    print(f"❌ Error: {e}")
