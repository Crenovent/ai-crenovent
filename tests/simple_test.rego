package test

import data.policies

# Test admin access (should always pass)
test_admin_access {
    policies.allow with input as {
        "user": "admin",
        "action": "read",
        "resource": "data",
        "environment": "production"
    }
}

# Test simple read access for alice
test_alice_read_data {
    policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "data",
        "environment": "production"
    }
}
