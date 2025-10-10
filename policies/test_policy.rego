# Simple test policy to trigger workflow
package test

# This is a simple test policy
default allow := false

allow {
    input.user == "test"
}
