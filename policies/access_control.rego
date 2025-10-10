package policies

# Simple policy that always allows alice to read data
default allow := false

allow {
    input.user == "alice"
    input.action == "read"
    input.resource == "data"
}

allow {
    input.user == "admin"
}