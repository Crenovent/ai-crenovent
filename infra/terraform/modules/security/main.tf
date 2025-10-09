# ================================================================================
# Security Module - Key Vault, RBAC, and Security Policies
# ================================================================================

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.0.0"
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "security" {
  name     = "${var.name_prefix}-security-rg"
  location = var.location
  tags     = var.tags
}

# Key Vault
resource "azurerm_key_vault" "main" {
  count               = var.enable_key_vault ? 1 : 0
  name                = "${var.name_prefix}-kv"
  location            = azurerm_resource_group.security.location
  resource_group_name = azurerm_resource_group.security.name
  tenant_id           = var.tenant_id
  sku_name            = var.key_vault_sku
  tags                = var.tags

  enabled_for_disk_encryption     = true
  enabled_for_deployment          = true
  enabled_for_template_deployment = true
  enable_rbac_authorization       = true
  purge_protection_enabled        = var.enable_purge_protection
  soft_delete_retention_days      = var.soft_delete_retention_days

  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    ip_rules       = var.allowed_ip_ranges
  }

  depends_on = [azurerm_resource_group.security]
}

# Key Vault Access Policy for Current User
resource "azurerm_key_vault_access_policy" "current_user" {
  count        = var.enable_key_vault ? 1 : 0
  key_vault_id = azurerm_key_vault.main[0].id
  tenant_id    = var.tenant_id
  object_id    = var.current_user_object_id

  key_permissions = [
    "Get", "List", "Create", "Delete", "Update", "Import", "Backup", "Restore", "Recover"
  ]

  secret_permissions = [
    "Get", "List", "Set", "Delete", "Backup", "Restore", "Recover"
  ]

  certificate_permissions = [
    "Get", "List", "Create", "Delete", "Update", "Import", "Backup", "Restore", "Recover"
  ]
}

# Key Vault Access Policy for AKS
resource "azurerm_key_vault_access_policy" "aks" {
  count        = var.enable_key_vault && var.aks_managed_identity_id != "" ? 1 : 0
  key_vault_id = azurerm_key_vault.main[0].id
  tenant_id    = var.tenant_id
  object_id    = var.aks_managed_identity_id

  secret_permissions = [
    "Get", "List"
  ]
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "postgresql_password" {
  count        = var.enable_key_vault && var.postgresql_password != "" ? 1 : 0
  name         = "postgresql-password"
  value        = var.postgresql_password
  key_vault_id = azurerm_key_vault.main[0].id
  depends_on   = [azurerm_key_vault_access_policy.current_user]
}

resource "azurerm_key_vault_secret" "redis_password" {
  count        = var.enable_key_vault && var.redis_password != "" ? 1 : 0
  name         = "redis-password"
  value        = var.redis_password
  key_vault_id = azurerm_key_vault.main[0].id
  depends_on   = [azurerm_key_vault_access_policy.current_user]
}

resource "azurerm_key_vault_secret" "jwt_secret" {
  count        = var.enable_key_vault && var.jwt_secret != "" ? 1 : 0
  name         = "jwt-secret"
  value        = var.jwt_secret
  key_vault_id = azurerm_key_vault.main[0].id
  depends_on   = [azurerm_key_vault_access_policy.current_user]
}

# Azure Policy Assignment for Security
resource "azurerm_policy_assignment" "security_policies" {
  count                = var.enable_policy_assignment ? length(var.policy_definitions) : 0
  name                 = "${var.name_prefix}-policy-${count.index}"
  scope                = var.policy_scope
  policy_definition_id = var.policy_definitions[count.index]
  description          = "Security policy assignment for RevAI"
  display_name         = "RevAI Security Policy ${count.index + 1}"
}

# Security Center Contact
resource "azurerm_security_center_contact" "main" {
  count               = var.enable_security_center ? 1 : 0
  email               = var.security_contact_email
  phone               = var.security_contact_phone
  alert_notifications = true
  alerts_to_admins    = true
}

# Security Center Subscription Pricing
resource "azurerm_security_center_subscription_pricing" "main" {
  count         = var.enable_security_center ? 1 : 0
  tier          = var.security_center_tier
  resource_type = "VirtualMachines"
}

# Outputs
output "key_vault_id" {
  description = "The ID of the Key Vault"
  value       = var.enable_key_vault ? azurerm_key_vault.main[0].id : null
}

output "key_vault_uri" {
  description = "The URI of the Key Vault"
  value       = var.enable_key_vault ? azurerm_key_vault.main[0].vault_uri : null
}

output "key_vault_name" {
  description = "The name of the Key Vault"
  value       = var.enable_key_vault ? azurerm_key_vault.main[0].name : null
}

output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.security.name
}
