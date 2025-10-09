# ================================================================================
# RevAI Infrastructure Deployment Script (PowerShell)
# ================================================================================

param(
    [Parameter(Position=0)]
    [string]$Environment = "dev",
    
    [Parameter(Position=1)]
    [bool]$DeployInfra = $true,
    
    [Parameter(Position=2)]
    [bool]$DeployApps = $true,
    
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TerraformDir = Join-Path $ScriptDir "terraform"
$HelmDir = Join-Path $ScriptDir "helm"

# Functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $missingTools = @()
    
    if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
        $missingTools += "terraform"
    }
    
    if (-not (Get-Command helm -ErrorAction SilentlyContinue)) {
        $missingTools += "helm"
    }
    
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        $missingTools += "azure-cli"
    }
    
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        $missingTools += "kubectl"
    }
    
    if ($missingTools.Count -gt 0) {
        Write-Error "Missing required tools: $($missingTools -join ', ')"
        Write-Info "Please install the missing tools and try again."
        exit 1
    }
    
    Write-Success "All prerequisites are installed"
}

# Check Azure authentication
function Test-AzureAuth {
    Write-Info "Checking Azure authentication..."
    
    try {
        $account = az account show --query id -o tsv 2>$null
        if (-not $account) {
            throw "Not logged in to Azure CLI"
        }
        Write-Success "Authenticated to Azure subscription: $account"
    }
    catch {
        Write-Error "Not logged in to Azure CLI"
        Write-Info "Please run 'az login' and try again"
        exit 1
    }
}

# Deploy infrastructure
function Deploy-Infrastructure {
    param([string]$Environment)
    
    $terraformDir = Join-Path $TerraformDir "environments" $Environment
    
    Write-Info "Deploying infrastructure for environment: $Environment"
    
    if (-not (Test-Path $terraformDir)) {
        Write-Error "Environment directory not found: $terraformDir"
        exit 1
    }
    
    Set-Location $terraformDir
    
    # Initialize Terraform
    Write-Info "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    Write-Info "Planning Terraform deployment..."
    terraform plan -var-file="$Environment.tfvars" -out=tfplan
    
    # Apply deployment
    Write-Info "Applying Terraform deployment..."
    terraform apply tfplan
    
    Write-Success "Infrastructure deployed successfully"
}

# Deploy applications
function Deploy-Applications {
    param([string]$Environment)
    
    $resourceGroup = "rg-revai-$Environment"
    $aksName = "revai-$Environment-aks"
    
    Write-Info "Deploying applications for environment: $Environment"
    
    # Get AKS credentials
    Write-Info "Getting AKS credentials..."
    az aks get-credentials --resource-group $resourceGroup --name $aksName --overwrite-existing
    
    # Create namespace
    Write-Info "Creating namespace..."
    kubectl create namespace revai --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy orchestrator
    Write-Info "Deploying orchestrator..."
    helm upgrade --install orchestrator (Join-Path $HelmDir "orchestrator") `
        --namespace revai `
        --set image.tag=latest `
        --set secrets.postgresql_password=$env:POSTGRESQL_PASSWORD `
        --set secrets.redis_password=$env:REDIS_PASSWORD `
        --set secrets.jwt_secret=$env:JWT_SECRET
    
    # Deploy evidence worker
    Write-Info "Deploying evidence worker..."
    helm upgrade --install evidence-worker (Join-Path $HelmDir "evidence-worker") `
        --namespace revai `
        --set image.tag=latest `
        --set secrets.postgresql_password=$env:POSTGRESQL_PASSWORD `
        --set secrets.redis_password=$env:REDIS_PASSWORD `
        --set secrets.jwt_secret=$env:JWT_SECRET
    
    Write-Success "Applications deployed successfully"
}

# Verify deployment
function Test-Deployment {
    param([string]$Environment)
    
    Write-Info "Verifying deployment for environment: $Environment"
    
    # Check pods
    Write-Info "Checking pod status..."
    kubectl get pods -n revai
    
    # Check services
    Write-Info "Checking service status..."
    kubectl get services -n revai
    
    # Check ingress
    Write-Info "Checking ingress status..."
    kubectl get ingress -n revai
    
    Write-Success "Deployment verification completed"
}

# Usage function
function Show-Usage {
    Write-Host "Usage: .\deploy.ps1 [environment] [deploy_infra] [deploy_apps]"
    Write-Host ""
    Write-Host "Arguments:"
    Write-Host "  environment    Environment to deploy (dev, staging, prod) [default: dev]"
    Write-Host "  deploy_infra   Deploy infrastructure (true, false) [default: true]"
    Write-Host "  deploy_apps    Deploy applications (true, false) [default: true]"
    Write-Host ""
    Write-Host "Environment Variables:"
    Write-Host "  POSTGRESQL_PASSWORD    PostgreSQL password"
    Write-Host "  REDIS_PASSWORD         Redis password"
    Write-Host "  JWT_SECRET             JWT secret key"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 dev true true"
    Write-Host "  .\deploy.ps1 staging false true"
    Write-Host "  .\deploy.ps1 prod true false"
}

# Main function
function Main {
    Write-Info "Starting RevAI infrastructure deployment"
    Write-Info "Environment: $Environment"
    Write-Info "Deploy Infrastructure: $DeployInfra"
    Write-Info "Deploy Applications: $DeployApps"
    
    # Check prerequisites
    Test-Prerequisites
    
    # Check Azure authentication
    Test-AzureAuth
    
    # Deploy infrastructure
    if ($DeployInfra) {
        Deploy-Infrastructure $Environment
    }
    
    # Deploy applications
    if ($DeployApps) {
        Deploy-Applications $Environment
    }
    
    # Verify deployment
    Test-Deployment $Environment
    
    Write-Success "RevAI infrastructure deployment completed successfully!"
}

# Check if help is requested
if ($Help) {
    Show-Usage
    exit 0
}

# Check required environment variables
if (-not $env:POSTGRESQL_PASSWORD -or -not $env:REDIS_PASSWORD -or -not $env:JWT_SECRET) {
    Write-Error "Required environment variables not set"
    Write-Info "Please set the following environment variables:"
    Write-Info "  POSTGRESQL_PASSWORD"
    Write-Info "  REDIS_PASSWORD"
    Write-Info "  JWT_SECRET"
    exit 1
}

# Run main function
Main
