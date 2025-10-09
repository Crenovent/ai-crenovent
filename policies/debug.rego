package policies

# Simple policy for debugging
default allow := false

# Allow admin access
allow {
    input.user == "admin"
}

# Allow alice to read data
allow {
    input.user == "alice"
    input.action == "read"
    input.resource == "data"
}
