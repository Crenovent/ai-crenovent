# ================================================================================
# Key Vault Test Service Deployment Script (PowerShell)
# ================================================================================

param(
    [string]$Namespace = "revai-staging",
    [string]$HelmChartPath = "./infra/helm/keyvault-test",
    [string]$KeyVaultName = "kv-revai-staging",
    [switch]$Cleanup
)

# Configuration
$ErrorActionPreference = "Stop"

# Functions
function Write-LogInfo {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-LogWarn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-LogError {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-LogInfo "Checking prerequisites..."
    
    # Check if kubectl is available
    try {
        kubectl version --client | Out-Null
    }
    catch {
        Write-LogError "kubectl is not installed or not in PATH"
        exit 1
    }
    
    # Check if helm is available
    try {
        helm version | Out-Null
    }
    catch {
        Write-LogError "helm is not installed or not in PATH"
        exit 1
    }
    
    # Check if az is available
    try {
        az version | Out-Null
    }
    catch {
        Write-LogError "Azure CLI is not installed or not in PATH"
        exit 1
    }
    
    Write-LogInfo "Prerequisites check passed"
}

# Create namespace if it doesn't exist
function New-NamespaceIfNotExists {
    param([string]$Namespace)
    
    Write-LogInfo "Creating namespace: $Namespace"
    
    try {
        kubectl get namespace $Namespace | Out-Null
        Write-LogInfo "Namespace $Namespace already exists"
    }
    catch {
        kubectl create namespace $Namespace
        Write-LogInfo "Namespace $Namespace created"
    }
}

# Get managed identity client ID
function Get-ManagedIdentityClientId {
    Write-LogInfo "Getting managed identity client ID..."
    
    try {
        $ManagedIdentityClientId = az identity list --resource-group rg-revai-staging --query "[?contains(name, 'aks')].clientId" -o tsv | Select-Object -First 1
        
        if ([string]::IsNullOrEmpty($ManagedIdentityClientId)) {
            Write-LogWarn "Could not find managed identity client ID automatically"
            $ManagedIdentityClientId = Read-Host "Enter managed identity client ID"
        }
        
        Write-LogInfo "Using managed identity client ID: $ManagedIdentityClientId"
        return $ManagedIdentityClientId
    }
    catch {
        Write-LogError "Failed to get managed identity client ID"
        exit 1
    }
}

# Deploy the test service
function Deploy-TestService {
    param(
        [string]$HelmChartPath,
        [string]$Namespace,
        [string]$ManagedIdentityClientId,
        [string]$KeyVaultName
    )
    
    Write-LogInfo "Deploying Key Vault test service..."
    
    helm upgrade --install kv-test $HelmChartPath `
        --namespace $Namespace `
        --set "managedIdentity.clientId=$ManagedIdentityClientId" `
        --set "keyvault.name=$KeyVaultName" `
        --wait `
        --timeout=5m
    
    Write-LogInfo "Key Vault test service deployed successfully"
}

# Run verification job
function Start-VerificationJob {
    param([string]$Namespace)
    
    Write-LogInfo "Running verification job..."
    
    # Wait for job to complete
    kubectl wait --for=condition=complete job/kv-test-job -n $Namespace --timeout=300s
    
    # Get job logs
    Write-LogInfo "Verification job logs:"
    kubectl logs job/kv-test-job -n $Namespace
    
    # Check job status
    $JobStatus = kubectl get job kv-test-job -n $Namespace -o jsonpath='{.status.conditions[0].type}'
    
    if ($JobStatus -eq "Complete") {
        Write-LogInfo "✅ Verification job completed successfully"
    }
    else {
        Write-LogError "❌ Verification job failed"
        exit 1
    }
}

# Check audit logs
function Get-AuditLogs {
    Write-LogInfo "Checking audit logs..."
    
    try {
        $StartTime = (Get-Date).AddHours(-1).ToString("yyyy-MM-ddTHH:mm:ss")
        
        $AuditLogs = az monitor activity-log list `
            --resource-group rg-revai-staging `
            --resource-provider Microsoft.KeyVault `
            --start-time $StartTime `
            --query "[].{time:eventTimestamp, operation:operationName.value, caller:caller}" `
            -o table
        
        if ($AuditLogs) {
            Write-LogInfo "Recent Key Vault audit logs:"
            Write-Host $AuditLogs
        }
        else {
            Write-LogWarn "No recent audit logs found"
        }
    }
    catch {
        Write-LogWarn "Failed to retrieve audit logs: $_"
    }
}

# Cleanup function
function Remove-TestResources {
    param([string]$Namespace)
    
    Write-LogInfo "Cleaning up test resources..."
    
    try {
        helm uninstall kv-test -n $Namespace
    }
    catch {
        Write-LogWarn "Failed to uninstall helm chart: $_"
    }
    
    try {
        kubectl delete job kv-test-job -n $Namespace
    }
    catch {
        Write-LogWarn "Failed to delete job: $_"
    }
    
    Write-LogInfo "Cleanup completed"
}

# Main execution
function Start-Main {
    Write-LogInfo "Starting Key Vault test service deployment..."
    
    Test-Prerequisites
    New-NamespaceIfNotExists -Namespace $Namespace
    $ManagedIdentityClientId = Get-ManagedIdentityClientId
    Deploy-TestService -HelmChartPath $HelmChartPath -Namespace $Namespace -ManagedIdentityClientId $ManagedIdentityClientId -KeyVaultName $KeyVaultName
    Start-VerificationJob -Namespace $Namespace
    Get-AuditLogs
    
    Write-LogInfo "🎉 Key Vault test service deployment and verification completed successfully!"
    
    if (-not $Cleanup) {
        $CleanupChoice = Read-Host "Do you want to cleanup test resources? (y/N)"
        if ($CleanupChoice -match "^[Yy]$") {
            Remove-TestResources -Namespace $Namespace
        }
        else {
            Write-LogInfo "Test resources left running for further inspection"
        }
    }
    else {
        Remove-TestResources -Namespace $Namespace
    }
}

# Run main function
try {
    Start-Main
}
catch {
    Write-LogError "Script failed: $_"
    exit 1
}
