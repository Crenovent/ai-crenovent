package test

import data.policies

# Test valid user read access
test_allow_valid_user_read {
    policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "data",
        "environment": "production"
    }
}

# Test admin access
test_allow_admin_access {
    policies.allow with input as {
        "user": "admin",
        "action": "write",
        "resource": "any-resource",
        "environment": "production"
    }
}

# Test invalid user access
test_deny_invalid_user {
    not policies.allow with input as {
        "user": "eve",
        "action": "read",
        "resource": "data",
        "environment": "production"
    }
}

# Test sensitive resource access
test_deny_sensitive_resource {
    not policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "sensitive-data",
        "environment": "production"
    }
}

# Test development environment access
test_allow_development_access {
    policies.allow with input as {
        "user": "alice",
        "action": "read",
        "resource": "data",
        "environment": "development"
    }
}

# Test write permissions
test_allow_write_permission {
    policies.allow with input as {
        "user": "alice",
        "action": "write",
        "resource": "data",
        "environment": "production"
    }
}

# Test deny write for non-write users
test_deny_write_for_readonly_user {
    not policies.allow with input as {
        "user": "bob",
        "action": "write",
        "resource": "data",
        "environment": "production"
    }
}
