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
    valid_users[input.user]
    readable_resources[input.resource]
}

# Allow write access for specific users
allow {
    input.action == "write"
    write_users[input.user]
    writable_resources[input.resource]
    input.environment == "production"
}

# Valid users set
valid_users := {
    "alice",
    "bob",
    "charlie"
}

# Users with write permissions set
write_users := {
    "alice",
    "admin"
}

# Readable resources set
readable_resources := {
    "data",
    "logs",
    "metrics"
}

# Writable resources set
writable_resources := {
    "data",
    "config"
}

# Deny access to sensitive resources
deny {
    input.resource == "sensitive-data"
    input.user != "admin"
}

# Environment-based access control
allow {
    input.environment == "development"
    valid_users[input.user]
}

# Time-based access control (example)
allow {
    input.action == "read"
    valid_users[input.user]
    hour := time.clock(input.time)[0]
    hour >= 9
    hour <= 17
}
