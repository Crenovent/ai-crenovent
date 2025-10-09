# ================================================================================
# Staging Environment Configuration
# ================================================================================

terraform {
  required_version = ">= 1.9.2"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.0.0, < 5.0.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Local variables
locals {
  environment = "staging"
  location   = "West Europe"
  
  common_tags = {
    Environment = local.environment
    Project     = "RevAI"
    ManagedBy   = "Terraform"
    Owner       = "devops"
    CostCenter  = "engineering"
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-revai-${local.environment}"
  location = local.location
  tags     = local.common_tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-revai-${local.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 90
  tags                = local.common_tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "ai-revai-${local.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.common_tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "acrrevai${local.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = true
  tags                = local.common_tags
}

# Service Bus
resource "azurerm_servicebus_namespace" "main" {
  name                = "sb-revai-${local.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"
  capacity            = 2
  tags                = local.common_tags
}

# AKS Module
module "aks" {
  source = "../../modules/aks"
  
  name_prefix = "revai-${local.environment}"
  location    = local.location
  tags        = local.common_tags
  
  kubernetes_version = "1.28"
  node_count         = 3
  vm_size           = "Standard_D4s_v4"
  enable_auto_scaling = true
  min_count         = 2
  max_count         = 8
  enable_monitoring = true
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  
  vnet_address_space     = ["10.1.0.0/16"]
  subnet_address_prefixes = ["10.1.1.0/24"]
  pod_subnet_address_prefixes = ["10.1.2.0/24"]
  
  enable_app_node_pool = true
  app_vm_size         = "Standard_D8s_v4"
  app_node_count      = 2
  app_min_count       = 2
  app_max_count       = 15
}

# Storage Module
module "storage" {
  source = "../../modules/storage"
  
  name_prefix = "revai-${local.environment}"
  location    = local.location
  tags        = local.common_tags
  
  enable_postgresql = true
  postgresql_admin_username = "postgresadmin"
  postgresql_admin_password = var.postgresql_password
  postgresql_database_name = "revai_staging"
  postgresql_sku = "GP_Standard_D4s_v3"
  postgresql_storage_mb = 65536
  postgresql_backup_retention_days = 14
  postgresql_geo_redundant_backup = true
  postgresql_high_availability_mode = "ZoneRedundant"
  
  enable_redis = true
  redis_sku = "Standard"
  redis_capacity = 2
  redis_family = "C"
  
  enable_storage_account = true
  storage_account_tier = "Standard"
  storage_account_replication_type = "GRS"
}

# Security Module
module "security" {
  source = "../../modules/security"
  
  name_prefix = "revai-${local.environment}"
  location    = local.location
  tags        = local.common_tags
  
  enable_key_vault = true
  tenant_id = data.azurerm_client_config.current.tenant_id
  current_user_object_id = data.azurerm_client_config.current.object_id
  aks_managed_identity_id = module.aks.aks_cluster_identity[0].principal_id
  
  enable_purge_protection = true
  soft_delete_retention_days = 30
  
  postgresql_password = var.postgresql_password
  redis_password = var.redis_password
  jwt_secret = var.jwt_secret
  
  enable_policy_assignment = true
  policy_scope = azurerm_resource_group.main.id
  
  enable_security_center = true
}

# Data sources
data "azurerm_client_config" "current" {}

# Variables
variable "postgresql_password" {
  description = "PostgreSQL administrator password"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}

# Outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  description = "Name of the AKS cluster"
  value       = module.aks.aks_cluster_name
}

output "aks_cluster_fqdn" {
  description = "FQDN of the AKS cluster"
  value       = module.aks.aks_cluster_fqdn
}

output "container_registry_name" {
  description = "Name of the container registry"
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Login server of the container registry"
  value       = azurerm_container_registry.main.login_server
}

output "postgresql_server_fqdn" {
  description = "FQDN of the PostgreSQL server"
  value       = module.storage.postgresql_server_fqdn
}

output "redis_cache_hostname" {
  description = "Hostname of the Redis cache"
  value       = module.storage.redis_cache_hostname
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = module.security.key_vault_uri
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}
