package policies

# Default deny
default allow := false

# Policy version: v1.1 - Fixed syntax errors

# Allow access if user is admin
allow {
    input.user == "admin"
}

# Allow read access for valid users (explicitly exclude sensitive-data)
allow {
    input.action == "read"
    valid_users[input.user]
    readable_resources[input.resource]
    input.resource != "sensitive-data"
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

# Allow admin access to sensitive resources
allow {
    input.user == "admin"
    input.resource == "sensitive-data"
}

# Environment-based access control (only for development)
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
