package test

import data.policies

# Test admin access
test_admin {
    policies.allow with input as {
        "user": "admin",
        "action": "read",
        "resource": "data"
    }
}

# Test alice read access
test_alice_read {
    policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "data"
    }
}
