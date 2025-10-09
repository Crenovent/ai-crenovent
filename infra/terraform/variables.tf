# ================================================================================
# RevAI Infrastructure Variables
# ================================================================================

# Environment Configuration
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Primary Azure region"
  type        = string
  default     = "Central India"
}

variable "secondary_location" {
  description = "Secondary Azure region for DR"
  type        = string
  default     = "West Europe"
}

# Resource Naming
variable "owner" {
  description = "Resource owner/team"
  type        = string
  default     = "devops"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

variable "compliance_level" {
  description = "Compliance level (basic, enhanced, critical)"
  type        = string
  default     = "enhanced"
  validation {
    condition     = contains(["basic", "enhanced", "critical"], var.compliance_level)
    error_message = "Compliance level must be one of: basic, enhanced, critical."
  }
}

# Kubernetes Configuration
variable "aks_config" {
  description = "AKS cluster configuration"
  type = object({
    kubernetes_version = string
    node_count        = number
    vm_size           = string
    enable_auto_scaling = bool
    min_count         = number
    max_count         = number
    enable_monitoring = bool
  })
  default = {
    kubernetes_version = "1.28"
    node_count        = 3
    vm_size           = "Standard_D4s_v4"
    enable_auto_scaling = true
    min_count         = 2
    max_count         = 10
    enable_monitoring = true
  }
}

# Storage Configuration
variable "storage_config" {
  description = "Storage configuration"
  type = object({
    enable_postgresql = bool
    postgresql_tier   = string
    postgresql_sku    = string
    enable_redis      = bool
    redis_tier        = string
    redis_capacity    = number
  })
  default = {
    enable_postgresql = true
    postgresql_tier   = "GeneralPurpose"
    postgresql_sku    = "GP_Gen5_2"
    enable_redis      = true
    redis_tier        = "Standard"
    redis_capacity    = 1
  }
}

# Security Configuration
variable "security_config" {
  description = "Security configuration"
  type = object({
    enable_key_vault = bool
    enable_rbac      = bool
    enable_nsg       = bool
    allowed_ips      = list(string)
  })
  default = {
    enable_key_vault = true
    enable_rbac      = true
    enable_nsg       = true
    allowed_ips      = []
  }
}

# Monitoring Configuration
variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    enable_log_analytics = bool
    enable_app_insights  = bool
    retention_days       = number
    enable_alerts        = bool
  })
  default = {
    enable_log_analytics = true
    enable_app_insights  = true
    retention_days       = 30
    enable_alerts        = true
  }
}

# Networking Configuration
variable "networking_config" {
  description = "Networking configuration"
  type = object({
    vnet_address_space     = list(string)
    subnet_address_prefixes = list(string)
    enable_private_endpoints = bool
    enable_public_ip        = bool
  })
  default = {
    vnet_address_space     = ["10.0.0.0/16"]
    subnet_address_prefixes = ["10.0.1.0/24", "10.0.2.0/24"]
    enable_private_endpoints = true
    enable_public_ip        = false
  }
}

# Container Registry Configuration
variable "container_registry_config" {
  description = "Container registry configuration"
  type = object({
    enable_acr        = bool
    acr_sku           = string
    enable_admin_user  = bool
  })
  default = {
    enable_acr        = true
    acr_sku           = "Standard"
    enable_admin_user  = false
  }
}

# Service Bus Configuration
variable "servicebus_config" {
  description = "Service Bus configuration"
  type = object({
    enable_servicebus = bool
    sku               = string
    capacity          = number
  })
  default = {
    enable_servicebus = true
    sku               = "Standard"
    capacity          = 1
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for optional components"
  type = object({
    enable_cosmosdb     = bool
    enable_openai       = bool
    enable_cdn          = bool
    enable_waf          = bool
    enable_backup       = bool
  })
  default = {
    enable_cosmosdb     = false
    enable_openai       = false
    enable_cdn          = false
    enable_waf          = false
    enable_backup       = true
  }
}
