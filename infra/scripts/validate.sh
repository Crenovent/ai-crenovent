#!/bin/bash

# ================================================================================
# RevAI Infrastructure Validation Script
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

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    ((PASSED_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((FAILED_CHECKS++))
}

# Check function
check() {
    local description="$1"
    local command="$2"
    
    ((TOTAL_CHECKS++))
    log_info "Checking: $description"
    
    if eval "$command" >/dev/null 2>&1; then
        log_success "$description"
        return 0
    else
        log_error "$description"
        return 1
    fi
}

# Terraform validation
validate_terraform() {
    log_info "Validating Terraform configuration..."
    
    # Check Terraform format
    check "Terraform format check" "terraform fmt -check -recursive $TERRAFORM_DIR"
    
    # Check Terraform validation for each environment
    for env in dev staging prod; do
        local env_dir="${TERRAFORM_DIR}/environments/${env}"
        if [ -d "$env_dir" ]; then
            check "Terraform validation ($env)" "cd $env_dir && terraform init -backend=false && terraform validate"
        fi
    done
    
    # Check Terraform security
    if command -v tfsec &> /dev/null; then
        check "Terraform security scan" "tfsec $TERRAFORM_DIR"
    else
        log_warning "tfsec not installed, skipping security scan"
    fi
    
    # Check Terraform linting
    if command -v tflint &> /dev/null; then
        check "Terraform linting" "tflint $TERRAFORM_DIR"
    else
        log_warning "tflint not installed, skipping linting"
    fi
}

# Helm validation
validate_helm() {
    log_info "Validating Helm charts..."
    
    # Check Helm charts
    for chart in orchestrator evidence-worker; do
        local chart_dir="${HELM_DIR}/${chart}"
        if [ -d "$chart_dir" ]; then
            check "Helm lint ($chart)" "helm lint $chart_dir"
            check "Helm template ($chart)" "helm template $chart $chart_dir"
        fi
    done
}

# File structure validation
validate_structure() {
    log_info "Validating file structure..."
    
    # Check required directories
    local required_dirs=(
        "$TERRAFORM_DIR"
        "$TERRAFORM_DIR/modules"
        "$TERRAFORM_DIR/modules/aks"
        "$TERRAFORM_DIR/modules/storage"
        "$TERRAFORM_DIR/modules/security"
        "$TERRAFORM_DIR/environments"
        "$TERRAFORM_DIR/environments/dev"
        "$HELM_DIR"
        "$HELM_DIR/orchestrator"
        "$HELM_DIR/evidence-worker"
    )
    
    for dir in "${required_dirs[@]}"; do
        check "Directory exists: $(basename "$dir")" "[ -d '$dir' ]"
    done
    
    # Check required files
    local required_files=(
        "$TERRAFORM_DIR/main.tf"
        "$TERRAFORM_DIR/variables.tf"
        "$TERRAFORM_DIR/modules/aks/main.tf"
        "$TERRAFORM_DIR/modules/aks/variables.tf"
        "$TERRAFORM_DIR/modules/storage/main.tf"
        "$TERRAFORM_DIR/modules/storage/variables.tf"
        "$TERRAFORM_DIR/modules/security/main.tf"
        "$TERRAFORM_DIR/modules/security/variables.tf"
        "$TERRAFORM_DIR/environments/dev/main.tf"
        "$TERRAFORM_DIR/environments/dev/dev.tfvars"
        "$HELM_DIR/orchestrator/Chart.yaml"
        "$HELM_DIR/orchestrator/values.yaml"
        "$HELM_DIR/orchestrator/templates/deployment.yaml"
        "$HELM_DIR/orchestrator/templates/service.yaml"
        "$HELM_DIR/orchestrator/templates/_helpers.tpl"
        "$HELM_DIR/evidence-worker/Chart.yaml"
        "$HELM_DIR/evidence-worker/values.yaml"
    )
    
    for file in "${required_files[@]}"; do
        check "File exists: $(basename "$file")" "[ -f '$file' ]"
    done
}

# Documentation validation
validate_documentation() {
    log_info "Validating documentation..."
    
    local doc_files=(
        "README.md"
        "docs/contributing.md"
        "docs/deployment.md"
    )
    
    for file in "${doc_files[@]}"; do
        local file_path="${SCRIPT_DIR}/../$file"
        if [ -f "$file_path" ]; then
            check "Documentation exists: $file" "true"
        else
            log_warning "Documentation missing: $file"
        fi
    done
}

# Security validation
validate_security() {
    log_info "Validating security configuration..."
    
    # Check for hardcoded secrets
    check "No hardcoded secrets in Terraform" "! grep -r 'password.*=' $TERRAFORM_DIR --include='*.tf' | grep -v 'variable'"
    
    # Check for hardcoded secrets in Helm values
    check "No hardcoded secrets in Helm values" "! grep -r 'password.*:' $HELM_DIR --include='*.yaml' | grep -v 'secrets:'"
    
    # Check for proper security contexts in Helm templates
    check "Security contexts in Helm templates" "grep -r 'securityContext' $HELM_DIR/templates"
    
    # Check for non-root user configuration
    check "Non-root user configuration" "grep -r 'runAsNonRoot.*true' $HELM_DIR/templates"
}

# CI/CD validation
validate_cicd() {
    log_info "Validating CI/CD configuration..."
    
    local cicd_dir="${SCRIPT_DIR}/.github/workflows"
    if [ -d "$cicd_dir" ]; then
        check "CI/CD workflow exists" "[ -f '$cicd_dir/infrastructure.yml' ]"
        
        # Check for required CI/CD steps
        if [ -f "$cicd_dir/infrastructure.yml" ]; then
            check "Terraform plan in CI/CD" "grep -q 'terraform plan' '$cicd_dir/infrastructure.yml'"
            check "Helm lint in CI/CD" "grep -q 'helm lint' '$cicd_dir/infrastructure.yml'"
            check "Security scan in CI/CD" "grep -q 'tfsec' '$cicd_dir/infrastructure.yml'"
        fi
    else
        log_warning "CI/CD directory not found"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "Validation Summary"
    echo "=========================================="
    echo "Total checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        log_success "All validations passed!"
        exit 0
    else
        log_error "Some validations failed. Please fix the issues above."
        exit 1
    fi
}

# Main function
main() {
    log_info "Starting RevAI infrastructure validation"
    
    validate_structure
    validate_terraform
    validate_helm
    validate_documentation
    validate_security
    validate_cicd
    
    print_summary
}

# Usage function
usage() {
    echo "Usage: $0"
    echo ""
    echo "This script validates the RevAI infrastructure configuration."
    echo "It checks:"
    echo "  - File structure and required files"
    echo "  - Terraform configuration and security"
    echo "  - Helm charts and templates"
    echo "  - Documentation completeness"
    echo "  - Security best practices"
    echo "  - CI/CD configuration"
    echo ""
    echo "Prerequisites:"
    echo "  - terraform"
    echo "  - helm"
    echo "  - tfsec (optional)"
    echo "  - tflint (optional)"
}

# Check if help is requested
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"
