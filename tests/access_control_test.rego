package test

import data.policies

# Test alice read access - should pass
test_allow_valid_user_read {
    policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "data",
        "environment": "production"
    }
}

# Test admin access - should pass
test_allow_admin_access {
    policies.allow with input as {
        "user": "admin",
        "action": "write",
        "resource": "any-resource",
        "environment": "production"
    }
}