# ================================================================================
# Security Module Variables
# ================================================================================

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Key Vault Configuration
variable "enable_key_vault" {
  description = "Enable Key Vault"
  type        = bool
  default     = true
}

variable "key_vault_sku" {
  description = "Key Vault SKU"
  type        = string
  default     = "standard"
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
}

variable "current_user_object_id" {
  description = "Object ID of the current user"
  type        = string
}

variable "aks_managed_identity_id" {
  description = "Managed identity ID of the AKS cluster"
  type        = string
  default     = ""
}

variable "enable_purge_protection" {
  description = "Enable purge protection for Key Vault"
  type        = bool
  default     = true
}

variable "soft_delete_retention_days" {
  description = "Soft delete retention days for Key Vault"
  type        = number
  default     = 90
}

variable "allowed_ip_ranges" {
  description = "Allowed IP ranges for Key Vault access"
  type        = list(string)
  default     = []
}

# Secrets
variable "postgresql_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "jwt_secret" {
  description = "JWT secret"
  type        = string
  sensitive   = true
  default     = ""
}

# Policy Configuration
variable "enable_policy_assignment" {
  description = "Enable policy assignments"
  type        = bool
  default     = true
}

variable "policy_scope" {
  description = "Scope for policy assignments"
  type        = string
}

variable "policy_definitions" {
  description = "List of policy definition IDs to assign"
  type        = list(string)
  default     = []
}

# Security Center Configuration
variable "enable_security_center" {
  description = "Enable Security Center"
  type        = bool
  default     = true
}

variable "security_contact_email" {
  description = "Security contact email"
  type        = string
  default     = "security@crenovent.com"
}

variable "security_contact_phone" {
  description = "Security contact phone"
  type        = string
  default     = "+1-555-0123"
}

variable "security_center_tier" {
  description = "Security Center pricing tier"
  type        = string
  default     = "Standard"
  validation {
    condition     = contains(["Free", "Standard"], var.security_center_tier)
    error_message = "Security Center tier must be either Free or Standard."
  }
}
