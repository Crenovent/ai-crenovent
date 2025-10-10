# ================================================================================
# Key Vault Module Variables
# ================================================================================

variable "name_prefix" {
  description = "Name prefix for the Key Vault"
  type        = string
}

variable "location" {
  description = "Azure location for the Key Vault"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
}

variable "sku_name" {
  description = "SKU name for the Key Vault (standard or premium)"
  type        = string
  default     = "standard"
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Security Configuration
variable "enable_purge_protection" {
  description = "Enable purge protection for the Key Vault"
  type        = bool
  default     = true
}

variable "soft_delete_retention_days" {
  description = "Number of days to retain soft deleted items"
  type        = number
  default     = 90
}

variable "network_default_action" {
  description = "Default action for network access (Allow or Deny)"
  type        = string
  default     = "Deny"
}

# RBAC Configuration
variable "admin_object_ids" {
  description = "List of object IDs for Key Vault Administrators"
  type        = list(string)
  default     = []
}

variable "secrets_officer_object_ids" {
  description = "List of object IDs for Key Vault Secrets Officers"
  type        = list(string)
  default     = []
}

variable "secrets_user_object_ids" {
  description = "List of object IDs for Key Vault Secrets Users"
  type        = list(string)
  default     = []
}

# PKI Configuration
variable "enable_pki" {
  description = "Enable PKI certificate generation"
  type        = bool
  default     = true
}

variable "pki_subject" {
  description = "Subject for PKI certificates"
  type        = string
  default     = "CN=RevAI Root CA, O=Crenovent, C=US"
}

variable "pki_dns_names" {
  description = "DNS names for PKI certificates"
  type        = list(string)
  default     = ["revai.crenovent.com", "*.revai.crenovent.com"]
}

# Application Secrets
variable "application_secrets" {
  description = "Map of application secrets to store in Key Vault"
  type        = map(string)
  default     = {}
}

# Integration Configuration
variable "enable_aks_integration" {
  description = "Enable AKS integration with Key Vault"
  type        = bool
  default     = true
}

variable "aks_managed_identity_id" {
  description = "Managed identity ID for AKS integration"
  type        = string
  default     = ""
}

variable "enable_diagnostics" {
  description = "Enable diagnostic logging for Key Vault"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for diagnostics"
  type        = string
  default     = ""
}
