package policies

# Default deny
default allow := false

# Allow access if user is admin
allow {
    input.user == "admin"
}

# Allow read access for valid users
allow {
    input.action == "read"
    input.user in valid_users
    input.resource in readable_resources
}

# Allow write access for specific users
allow {
    input.action == "write"
    input.user in write_users
    input.resource in writable_resources
    input.environment == "production"
}

# Valid users list
valid_users := [
    "alice",
    "bob",
    "charlie"
]

# Users with write permissions
write_users := [
    "alice",
    "admin"
]

# Readable resources
readable_resources := [
    "data",
    "logs",
    "metrics"
]

# Writable resources
writable_resources := [
    "data",
    "config"
]

# Deny access to sensitive resources
deny {
    input.resource == "sensitive-data"
    input.user != "admin"
}

# Environment-based access control
allow {
    input.environment == "development"
    input.user in valid_users
}

# Time-based access control (example)
allow {
    input.action == "read"
    input.user in valid_users
    hour := time.clock(input.time)[0]
    hour >= 9
    hour <= 17
}
