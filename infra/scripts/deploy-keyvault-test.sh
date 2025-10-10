#!/bin/bash
# ================================================================================
# Key Vault Test Service Deployment Script
# ================================================================================

set -e

# Configuration
NAMESPACE="revai-staging"
HELM_CHART_PATH="./infra/helm/keyvault-test"
KEYVAULT_NAME="kv-revai-staging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if az is available
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed or not in PATH"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_info "Namespace $NAMESPACE created"
    fi
}

# Get managed identity client ID
get_managed_identity_client_id() {
    log_info "Getting managed identity client ID..."
    
    # This would typically come from Terraform output or Azure CLI
    # For now, we'll use a placeholder that needs to be updated
    MANAGED_IDENTITY_CLIENT_ID=$(az identity list --resource-group rg-revai-staging --query "[?contains(name, 'aks')].clientId" -o tsv | head -1)
    
    if [ -z "$MANAGED_IDENTITY_CLIENT_ID" ]; then
        log_warn "Could not find managed identity client ID automatically"
        log_warn "Please set MANAGED_IDENTITY_CLIENT_ID environment variable"
        read -p "Enter managed identity client ID: " MANAGED_IDENTITY_CLIENT_ID
    fi
    
    log_info "Using managed identity client ID: $MANAGED_IDENTITY_CLIENT_ID"
}

# Deploy the test service
deploy_test_service() {
    log_info "Deploying Key Vault test service..."
    
    helm upgrade --install kv-test "$HELM_CHART_PATH" \
        --namespace "$NAMESPACE" \
        --set managedIdentity.clientId="$MANAGED_IDENTITY_CLIENT_ID" \
        --set keyvault.name="$KEYVAULT_NAME" \
        --wait \
        --timeout=5m
    
    log_info "Key Vault test service deployed successfully"
}

# Run verification job
run_verification_job() {
    log_info "Running verification job..."
    
    # Wait for job to complete
    kubectl wait --for=condition=complete job/kv-test-job -n "$NAMESPACE" --timeout=300s
    
    # Get job logs
    log_info "Verification job logs:"
    kubectl logs job/kv-test-job -n "$NAMESPACE"
    
    # Check job status
    JOB_STATUS=$(kubectl get job kv-test-job -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].type}')
    
    if [ "$JOB_STATUS" = "Complete" ]; then
        log_info "✅ Verification job completed successfully"
    else
        log_error "❌ Verification job failed"
        exit 1
    fi
}

# Check audit logs
check_audit_logs() {
    log_info "Checking audit logs..."
    
    # Get recent audit logs from Key Vault
    AUDIT_LOGS=$(az monitor activity-log list \
        --resource-group rg-revai-staging \
        --resource-provider Microsoft.KeyVault \
        --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
        --query "[].{time:eventTimestamp, operation:operationName.value, caller:caller}" \
        -o table)
    
    if [ -n "$AUDIT_LOGS" ]; then
        log_info "Recent Key Vault audit logs:"
        echo "$AUDIT_LOGS"
    else
        log_warn "No recent audit logs found"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test resources..."
    
    helm uninstall kv-test -n "$NAMESPACE" || true
    kubectl delete job kv-test-job -n "$NAMESPACE" || true
    
    log_info "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting Key Vault test service deployment..."
    
    check_prerequisites
    create_namespace
    get_managed_identity_client_id
    deploy_test_service
    run_verification_job
    check_audit_logs
    
    log_info "🎉 Key Vault test service deployment and verification completed successfully!"
    
    # Ask if user wants to cleanup
    read -p "Do you want to cleanup test resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    else
        log_info "Test resources left running for further inspection"
    fi
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"
