# ================================================================================
# Azure Key Vault Module - Secrets Manager & PKI Key Store
# ================================================================================

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.0.0"
    }
    azapi = {
      source  = "Azure/azapi"
      version = ">= 1.0.0"
    }
  }
}

# Data sources
data "azurerm_client_config" "current" {}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = var.name_prefix
  location            = var.location
  resource_group_name = var.resource_group_name
  tenant_id           = var.tenant_id
  sku_name            = var.sku_name
  tags                = var.tags

  # Security settings
  enabled_for_disk_encryption     = true
  enabled_for_deployment          = true
  enabled_for_template_deployment = true
  enable_rbac_authorization       = true
  purge_protection_enabled        = var.enable_purge_protection
  soft_delete_retention_days      = var.soft_delete_retention_days

  # Network access
  network_acls {
    default_action = var.network_default_action
    bypass         = "AzureServices"
  }

  # Access policies (legacy - using RBAC instead)
  access_policy = []
}

# RBAC: Key Vault Administrator
resource "azurerm_role_assignment" "kv_admin" {
  count                = length(var.admin_object_ids)
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Administrator"
  principal_id         = var.admin_object_ids[count.index]
}

# RBAC: Key Vault Secrets Officer (for service principals)
resource "azurerm_role_assignment" "kv_secrets_officer" {
  count                = length(var.secrets_officer_object_ids)
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = var.secrets_officer_object_ids[count.index]
}

# RBAC: Key Vault Secrets User (for applications)
resource "azurerm_role_assignment" "kv_secrets_user" {
  count                = length(var.secrets_user_object_ids)
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = var.secrets_user_object_ids[count.index]
}

# Evidence Signing Key (RSA 2048)
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

  tags = merge(var.tags, {
    Purpose  = "Evidence Signing"
    Rotation = "Automatic"
  })
}

# PKI Root CA Certificate
resource "azurerm_key_vault_certificate" "root_ca" {
  count        = var.enable_pki ? 1 : 0
  name         = "root-ca"
  key_vault_id = azurerm_key_vault.main.id

  certificate_policy {
    issuer_parameters {
      name = "Self"
    }

    key_properties {
      exportable = true
      key_size   = 4096
      key_type   = "RSA"
      reuse_key  = false
    }

    lifetime_action {
      action {
        action_type = "AutoRenew"
      }

      trigger {
        days_before_expiry = 30
      }
    }

    secret_properties {
      content_type = "application/x-pkcs12"
    }

    x509_certificate_properties {
      key_usage = [
        "cRLSign",
        "dataEncipherment",
        "digitalSignature",
        "keyAgreement",
        "keyCertSign",
        "keyEncipherment",
      ]

      subject_alternative_names {
        dns_names = var.pki_dns_names
      }

      subject            = var.pki_subject
      validity_in_months = 60
    }
  }

  tags = merge(var.tags, {
    Purpose = "PKI Root CA"
    Type    = "Certificate Authority"
  })
}

# Application Secrets
resource "azurerm_key_vault_secret" "app_secrets" {
  for_each = var.application_secrets

  name         = each.key
  value        = each.value
  key_vault_id = azurerm_key_vault.main.id

  tags = merge(var.tags, {
    Application = "RevAI"
    Type        = "Application Secret"
  })
}

# Key Vault Diagnostic Settings
resource "azurerm_monitor_diagnostic_setting" "kv_diagnostics" {
  count                      = var.enable_diagnostics ? 1 : 0
  name                       = "${var.name_prefix}-diagnostics"
  target_resource_id         = azurerm_key_vault.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id

  enabled_log {
    category = "AuditEvent"
  }

  enabled_log {
    category = "AzurePolicyEvaluationDetails"
  }

  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

# Key Vault Access Policy for AKS
resource "azurerm_key_vault_access_policy" "aks_policy" {
  count        = var.enable_aks_integration ? 1 : 0
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = var.tenant_id
  object_id    = var.aks_managed_identity_id

  key_permissions = [
    "Get",
    "List",
  ]

  secret_permissions = [
    "Get",
    "List",
  ]

  certificate_permissions = [
    "Get",
    "List",
  ]
}

# Outputs
output "key_vault_id" {
  description = "The ID of the Key Vault"
  value       = azurerm_key_vault.main.id
}

output "key_vault_uri" {
  description = "The URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "key_vault_name" {
  description = "The name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "evidence_signing_key_id" {
  description = "The ID of the evidence signing key"
  value       = azurerm_key_vault_key.evidence_signing_key.id
}

output "evidence_signing_key_version" {
  description = "The version of the evidence signing key"
  value       = azurerm_key_vault_key.evidence_signing_key.version
}

output "root_ca_certificate_id" {
  description = "The ID of the root CA certificate"
  value       = var.enable_pki ? azurerm_key_vault_certificate.root_ca[0].id : null
}

output "root_ca_certificate_version" {
  description = "The version of the root CA certificate"
  value       = var.enable_pki ? azurerm_key_vault_certificate.root_ca[0].version : null
}
