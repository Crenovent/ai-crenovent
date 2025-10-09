#!/bin/bash

# ================================================================================
# RevAI Infrastructure Deployment Script
# ================================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/terraform"
HELM_DIR="${SCRIPT_DIR}/helm"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    fi
    
    if ! command -v az &> /dev/null; then
        missing_tools+=("azure-cli")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again."
        exit 1
    fi
    
    log_success "All prerequisites are installed"
}

# Check Azure authentication
check_azure_auth() {
    log_info "Checking Azure authentication..."
    
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure CLI"
        log_info "Please run 'az login' and try again"
        exit 1
    fi
    
    local subscription_id=$(az account show --query id -o tsv)
    log_success "Authenticated to Azure subscription: ${subscription_id}"
}

# Deploy infrastructure
deploy_infrastructure() {
    local environment=$1
    local terraform_dir="${TERRAFORM_DIR}/environments/${environment}"
    
    log_info "Deploying infrastructure for environment: ${environment}"
    
    if [ ! -d "$terraform_dir" ]; then
        log_error "Environment directory not found: ${terraform_dir}"
        exit 1
    fi
    
    cd "$terraform_dir"
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -var-file="${environment}.tfvars" -out=tfplan
    
    # Apply deployment
    log_info "Applying Terraform deployment..."
    terraform apply tfplan
    
    log_success "Infrastructure deployed successfully"
}

# Deploy applications
deploy_applications() {
    local environment=$1
    local resource_group="rg-revai-${environment}"
    local aks_name="revai-${environment}-aks"
    
    log_info "Deploying applications for environment: ${environment}"
    
    # Get AKS credentials
    log_info "Getting AKS credentials..."
    az aks get-credentials --resource-group "$resource_group" --name "$aks_name" --overwrite-existing
    
    # Create namespace
    log_info "Creating namespace..."
    kubectl create namespace revai --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy orchestrator
    log_info "Deploying orchestrator..."
    helm upgrade --install orchestrator "${HELM_DIR}/orchestrator" \
        --namespace revai \
        --set image.tag=latest \
        --set secrets.postgresql_password="${POSTGRESQL_PASSWORD}" \
        --set secrets.redis_password="${REDIS_PASSWORD}" \
        --set secrets.jwt_secret="${JWT_SECRET}"
    
    # Deploy evidence worker
    log_info "Deploying evidence worker..."
    helm upgrade --install evidence-worker "${HELM_DIR}/evidence-worker" \
        --namespace revai \
        --set image.tag=latest \
        --set secrets.postgresql_password="${POSTGRESQL_PASSWORD}" \
        --set secrets.redis_password="${REDIS_PASSWORD}" \
        --set secrets.jwt_secret="${JWT_SECRET}"
    
    log_success "Applications deployed successfully"
}

# Verify deployment
verify_deployment() {
    local environment=$1
    
    log_info "Verifying deployment for environment: ${environment}"
    
    # Check pods
    log_info "Checking pod status..."
    kubectl get pods -n revai
    
    # Check services
    log_info "Checking service status..."
    kubectl get services -n revai
    
    # Check ingress
    log_info "Checking ingress status..."
    kubectl get ingress -n revai
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

# Main function
main() {
    local environment=${1:-dev}
    local deploy_infra=${2:-true}
    local deploy_apps=${3:-true}
    
    log_info "Starting RevAI infrastructure deployment"
    log_info "Environment: ${environment}"
    log_info "Deploy Infrastructure: ${deploy_infra}"
    log_info "Deploy Applications: ${deploy_apps}"
    
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Check Azure authentication
    check_azure_auth
    
    # Deploy infrastructure
    if [ "$deploy_infra" = "true" ]; then
        deploy_infrastructure "$environment"
    fi
    
    # Deploy applications
    if [ "$deploy_apps" = "true" ]; then
        deploy_applications "$environment"
    fi
    
    # Verify deployment
    verify_deployment "$environment"
    
    log_success "RevAI infrastructure deployment completed successfully!"
}

# Usage function
usage() {
    echo "Usage: $0 [environment] [deploy_infra] [deploy_apps]"
    echo ""
    echo "Arguments:"
    echo "  environment    Environment to deploy (dev, staging, prod) [default: dev]"
    echo "  deploy_infra   Deploy infrastructure (true, false) [default: true]"
    echo "  deploy_apps    Deploy applications (true, false) [default: true]"
    echo ""
    echo "Environment Variables:"
    echo "  POSTGRESQL_PASSWORD    PostgreSQL password"
    echo "  REDIS_PASSWORD         Redis password"
    echo "  JWT_SECRET             JWT secret key"
    echo ""
    echo "Examples:"
    echo "  $0 dev true true"
    echo "  $0 staging false true"
    echo "  $0 prod true false"
}

# Check if help is requested
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# Check required environment variables
if [ -z "${POSTGRESQL_PASSWORD:-}" ] || [ -z "${REDIS_PASSWORD:-}" ] || [ -z "${JWT_SECRET:-}" ]; then
    log_error "Required environment variables not set"
    log_info "Please set the following environment variables:"
    log_info "  POSTGRESQL_PASSWORD"
    log_info "  REDIS_PASSWORD"
    log_info "  JWT_SECRET"
    exit 1
fi

# Run main function
main "$@"
