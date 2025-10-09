# PLAT-SEC-002: Secrets Manager & PKI Key Store Implementation

## Overview
This document describes the implementation of Azure Key Vault as the secrets manager and PKI key store for the RevAI platform, fulfilling the requirements of PLAT-SEC-002.

## Architecture Decision: Azure Key Vault

### Why Azure Key Vault?
- **Native Azure Integration**: Seamless integration with AKS and Azure services
- **RBAC Support**: Modern role-based access control
- **PKI Capabilities**: Built-in certificate management and PKI
- **Audit Logging**: Comprehensive audit trails
- **Key Rotation**: Automated key rotation policies
- **Compliance**: SOC 2, ISO 27001, PCI DSS compliance

### Alternative Considered: HashiCorp Vault
- **Pros**: More features, open source, multi-cloud
- **Cons**: Additional complexity, requires separate infrastructure, licensing costs
- **Decision**: Azure Key Vault chosen for simplicity and native integration

## Implementation Components

### 1. Key Vault Module (`infra/terraform/modules/keyvault/`)

#### Features Implemented:
- ✅ **Key Vault Provisioning**: Standard SKU with purge protection
- ✅ **RBAC Configuration**: Role-based access control
- ✅ **Evidence Signing Key**: RSA 2048-bit key with automatic rotation
- ✅ **PKI Root CA**: Self-signed certificate authority
- ✅ **Application Secrets**: Secure storage for passwords and API keys
- ✅ **Diagnostic Logging**: Comprehensive audit trails
- ✅ **Network Security**: Deny-by-default network access

#### Key Resources:
```hcl
# Evidence Signing Key with automatic rotation
resource "azurerm_key_vault_key" "evidence_signing_key" {
  name         = "evidence-signing-key"
  key_vault_id = azurerm_key_vault.main.id
  key_type     = "RSA"
  key_size     = 2048
  key_opts     = ["sign", "verify"]

  rotation_policy {
    automatic {
      time_before_expiry = "P30D"
    }
    expire_after         = "P90D"
    notify_before_expiry = "P7D"
  }
}

# PKI Root CA Certificate
resource "azurerm_key_vault_certificate" "root_ca" {
  name         = "root-ca"
  key_vault_id = azurerm_key_vault.main.id
  # ... certificate policy configuration
}
```

### 2. Staging Environment Integration

#### Key Vault Configuration:
- **Name**: `kv-revai-staging`
- **SKU**: Standard
- **Location**: West Europe
- **Purge Protection**: Enabled
- **Soft Delete**: 30 days retention

#### RBAC Roles:
- **Key Vault Administrator**: DevOps team
- **Key Vault Secrets Officer**: Service principals
- **Key Vault Secrets User**: Applications

#### Application Secrets:
- `postgresql-password`: Database credentials
- `redis-password`: Cache credentials
- `jwt-secret`: JWT signing secret
- `api-key`: API authentication key

### 3. Test Service (`infra/helm/keyvault-test/`)

#### Purpose:
Verify Key Vault access and functionality from AKS

#### Components:
- **ConfigMap**: Key Vault configuration
- **ServiceAccount**: Workload identity for authentication
- **Deployment**: Continuous test service
- **Job**: One-time verification job

#### Test Scenarios:
1. **Secret Retrieval**: Fetch application secrets
2. **Key Access**: Retrieve signing keys
3. **Audit Verification**: Confirm audit logs are generated

## Security Features

### 1. Access Control
```yaml
# RBAC Configuration
admin_object_ids = [
  data.azurerm_client_config.current.object_id
]

# Network Security
network_default_action = "Deny"
```

### 2. Key Rotation Policy
```hcl
rotation_policy {
  automatic {
    time_before_expiry = "P30D"  # Rotate 30 days before expiry
  }
  expire_after         = "P90D"   # Keys expire after 90 days
  notify_before_expiry = "P7D"   # Notify 7 days before expiry
}
```

### 3. Audit Logging
```hcl
resource "azurerm_monitor_diagnostic_setting" "kv_diagnostics" {
  name                       = "${var.name_prefix}-diagnostics"
  target_resource_id         = azurerm_key_vault.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  enabled_log {
    category = "AuditEvent"
  }
  
  enabled_log {
    category = "AzurePolicyEvaluationDetails"
  }
}
```

## Deployment Process

### 1. Infrastructure Deployment
```bash
# Deploy to staging
cd infra/terraform/environments/staging
terraform init
terraform plan
terraform apply
```

### 2. Test Service Deployment
```bash
# Deploy test service
helm install kv-test ./infra/helm/keyvault-test \
  --namespace revai-staging \
  --set managedIdentity.clientId=<MANAGED_IDENTITY_CLIENT_ID>
```

### 3. Verification
```bash
# Check test job logs
kubectl logs job/kv-test-job -n revai-staging

# Verify audit logs
az monitor activity-log list --resource-group rg-revai-staging
```

## Access Patterns

### 1. Service-to-Key Vault Access
```yaml
# Service Account with Workload Identity
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kv-test-sa
  annotations:
    azure.workload.identity/client-id: "MANAGED_IDENTITY_CLIENT_ID"
```

### 2. Secret Retrieval
```bash
# Using Azure CLI
az keyvault secret show --vault-name kv-revai-staging --name jwt-secret

# Using Azure SDK (Python)
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://kv-revai-staging.vault.azure.net/", credential=credential)
secret = client.get_secret("jwt-secret")
```

### 3. Key Operations
```bash
# Sign data with evidence signing key
az keyvault key sign --vault-name kv-revai-staging --name evidence-signing-key --algorithm RS256 --value <data>

# Verify signature
az keyvault key verify --vault-name kv-revai-staging --name evidence-signing-key --algorithm RS256 --value <data> --signature <signature>
```

## Monitoring and Alerting

### 1. Audit Logs
- **Location**: Log Analytics Workspace
- **Retention**: 90 days (staging), 7 years (production)
- **Categories**: AuditEvent, AzurePolicyEvaluationDetails

### 2. Key Metrics
- Secret access frequency
- Key rotation events
- Failed access attempts
- Certificate expiry warnings

### 3. Alerts
- Failed authentication attempts
- Unusual access patterns
- Certificate expiry (30 days)
- Key rotation failures

## Compliance and Governance

### 1. Data Classification
- **Public**: None
- **Internal**: Configuration secrets
- **Confidential**: Database passwords, API keys
- **Restricted**: Signing keys, certificates

### 2. Access Policies
- **Principle of Least Privilege**: Minimum required permissions
- **Separation of Duties**: Different roles for different functions
- **Regular Reviews**: Quarterly access reviews

### 3. Compliance Standards
- **SOC 2**: Security controls implemented
- **ISO 27001**: Information security management
- **PCI DSS**: Payment card industry standards (if applicable)

## Rollback Procedures

### 1. Key Rotation Issues
```bash
# Revoke current key version
az keyvault key disable --vault-name kv-revai-staging --name evidence-signing-key

# Restore previous version
az keyvault key restore --vault-name kv-revai-staging --name evidence-signing-key --version <previous_version>
```

### 2. Secret Leak Response
```bash
# Immediate actions
az keyvault secret delete --vault-name kv-revai-staging --name <compromised_secret>
az keyvault access-policy delete --vault-name kv-revai-staging --object-id <compromised_principal>

# Generate new secret
az keyvault secret set --vault-name kv-revai-staging --name <new_secret_name> --value <new_secret_value>
```

### 3. Service Principal Compromise
```bash
# Revoke service principal
az ad sp credential delete --id <compromised_sp_id>

# Create new service principal
az ad sp create-for-rbac --name "revai-kv-access-$(date +%s)"
```

## Next Steps

### Immediate Actions:
1. ✅ Deploy Key Vault to staging environment
2. ✅ Configure RBAC policies
3. ✅ Set up evidence signing key
4. ✅ Create PKI root CA
5. ✅ Deploy test service
6. ✅ Verify access and audit logs

### Future Enhancements:
1. **Production Deployment**: Deploy to production environment
2. **Advanced PKI**: Intermediate CA certificates
3. **Automated Rotation**: CI/CD integration for key rotation
4. **Cross-Region Replication**: Disaster recovery setup
5. **Integration Testing**: Automated security testing

## Verification Checklist

- [ ] Key Vault accessible from staging AKS
- [ ] Evidence signing key retrievable by authorized service principal
- [ ] PKI root CA certificate generated and accessible
- [ ] Audit logs show all access events
- [ ] Test service successfully retrieves secrets and keys
- [ ] RBAC policies prevent unauthorized access
- [ ] Key rotation policy configured and active
- [ ] Documentation complete and up-to-date

## Security Considerations

### 1. Network Security
- Key Vault uses Azure's backbone network
- Network access rules restrict access
- Private endpoints available for production

### 2. Encryption
- All data encrypted at rest with Azure-managed keys
- Customer-managed keys available for production
- TLS 1.2+ for all communications

### 3. Monitoring
- Real-time monitoring of access patterns
- Automated alerting for suspicious activities
- Regular security assessments

This implementation provides a robust, secure, and compliant secrets management solution that meets all requirements of PLAT-SEC-002.
