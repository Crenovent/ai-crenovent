package policies

# Simple policy for testing
default allow := false

# Allow admin access
allow {
    input.user == "admin"
}

# Allow read access for valid users
allow {
    input.action == "read"
    input.user == "alice"
}
