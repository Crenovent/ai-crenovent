# ================================================================================
# Storage Module Variables
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

# PostgreSQL Configuration
variable "enable_postgresql" {
  description = "Enable PostgreSQL database"
  type        = bool
  default     = true
}

variable "postgresql_admin_username" {
  description = "PostgreSQL administrator username"
  type        = string
  default     = "postgresadmin"
}

variable "postgresql_admin_password" {
  description = "PostgreSQL administrator password"
  type        = string
  sensitive   = true
}

variable "postgresql_database_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "revai"
}

variable "postgresql_sku" {
  description = "PostgreSQL SKU"
  type        = string
  default     = "GP_Standard_D2s_v3"
}

variable "postgresql_storage_mb" {
  description = "PostgreSQL storage in MB"
  type        = number
  default     = 32768
}

variable "postgresql_backup_retention_days" {
  description = "PostgreSQL backup retention days"
  type        = number
  default     = 7
}

variable "postgresql_geo_redundant_backup" {
  description = "Enable geo-redundant backup"
  type        = bool
  default     = false
}

variable "postgresql_high_availability_mode" {
  description = "PostgreSQL high availability mode"
  type        = string
  default     = "Disabled"
  validation {
    condition     = contains(["Disabled", "ZoneRedundant", "SameZone"], var.postgresql_high_availability_mode)
    error_message = "High availability mode must be one of: Disabled, ZoneRedundant, SameZone."
  }
}

# Redis Configuration
variable "enable_redis" {
  description = "Enable Redis cache"
  type        = bool
  default     = true
}

variable "redis_sku" {
  description = "Redis SKU"
  type        = string
  default     = "Standard"
}

variable "redis_capacity" {
  description = "Redis capacity"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis family"
  type        = string
  default     = "C"
}

# Storage Account Configuration
variable "enable_storage_account" {
  description = "Enable storage account"
  type        = bool
  default     = true
}

variable "storage_account_tier" {
  description = "Storage account tier"
  type        = string
  default     = "Standard"
}

variable "storage_account_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"
}
