# ================================================================================
# Production Environment Configuration
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
  environment = "prod"
  primary_location   = "Central India"
  secondary_location = "West Europe"
  
  common_tags = {
    Environment = local.environment
    Project     = "RevAI"
    ManagedBy   = "Terraform"
    Owner       = "devops"
    CostCenter  = "engineering"
    Compliance  = "SOC2-ISO27001"
  }
}

# Primary Resource Group (India)
resource "azurerm_resource_group" "primary" {
  name     = "rg-revai-${local.environment}-primary"
  location = local.primary_location
  tags     = local.common_tags
}

# Secondary Resource Group (Europe)
resource "azurerm_resource_group" "secondary" {
  name     = "rg-revai-${local.environment}-secondary"
  location = local.secondary_location
  tags     = local.common_tags
}

# Log Analytics Workspace (Primary)
resource "azurerm_log_analytics_workspace" "primary" {
  name                = "law-revai-${local.environment}-primary"
  location            = azurerm_resource_group.primary.location
  resource_group_name = azurerm_resource_group.primary.name
  sku                 = "PerGB2018"
  retention_in_days   = 2555  # 7 years for compliance
  tags                = local.common_tags
}

# Log Analytics Workspace (Secondary)
resource "azurerm_log_analytics_workspace" "secondary" {
  name                = "law-revai-${local.environment}-secondary"
  location            = azurerm_resource_group.secondary.location
  resource_group_name = azurerm_resource_group.secondary.name
  sku                 = "PerGB2018"
  retention_in_days   = 2555  # 7 years for compliance
  tags                = local.common_tags
}

# Application Insights (Primary)
resource "azurerm_application_insights" "primary" {
  name                = "ai-revai-${local.environment}-primary"
  location            = azurerm_resource_group.primary.location
  resource_group_name = azurerm_resource_group.primary.name
  workspace_id        = azurerm_log_analytics_workspace.primary.id
  application_type    = "web"
  tags                = local.common_tags
}

# Application Insights (Secondary)
resource "azurerm_application_insights" "secondary" {
  name                = "ai-revai-${local.environment}-secondary"
  location            = azurerm_resource_group.secondary.location
  resource_group_name = azurerm_resource_group.secondary.name
  workspace_id        = azurerm_log_analytics_workspace.secondary.id
  application_type    = "web"
  tags                = local.common_tags
}

# Container Registry (Primary)
resource "azurerm_container_registry" "primary" {
  name                = "acrrevai${local.environment}primary"
  resource_group_name = azurerm_resource_group.primary.name
  location            = azurerm_resource_group.primary.location
  sku                 = "Premium"
  admin_enabled       = false  # Use managed identity
  tags                = local.common_tags
}

# Container Registry (Secondary)
resource "azurerm_container_registry" "secondary" {
  name                = "acrrevai${local.environment}secondary"
  resource_group_name = azurerm_resource_group.secondary.name
  location            = azurerm_resource_group.secondary.location
  sku                 = "Premium"
  admin_enabled       = false  # Use managed identity
  tags                = local.common_tags
}

# Service Bus (Primary)
resource "azurerm_servicebus_namespace" "primary" {
  name                = "sb-revai-${local.environment}-primary"
  location            = azurerm_resource_group.primary.location
  resource_group_name = azurerm_resource_group.primary.name
  sku                 = "Premium"
  capacity            = 4
  tags                = local.common_tags
}

# Service Bus (Secondary)
resource "azurerm_servicebus_namespace" "secondary" {
  name                = "sb-revai-${local.environment}-secondary"
  location            = azurerm_resource_group.secondary.location
  resource_group_name = azurerm_resource_group.secondary.name
  sku                 = "Premium"
  capacity            = 4
  tags                = local.common_tags
}

# AKS Module (Primary)
module "aks_primary" {
  source = "../../modules/aks"
  
  name_prefix = "revai-${local.environment}-primary"
  location    = local.primary_location
  tags        = local.common_tags
  
  kubernetes_version = "1.28"
  node_count         = 5
  vm_size           = "Standard_D8s_v4"
  enable_auto_scaling = true
  min_count         = 3
  max_count         = 20
  enable_monitoring = true
  log_analytics_workspace_id = azurerm_log_analytics_workspace.primary.id
  
  vnet_address_space     = ["10.2.0.0/16"]
  subnet_address_prefixes = ["10.2.1.0/24"]
  pod_subnet_address_prefixes = ["10.2.2.0/24"]
  
  enable_app_node_pool = true
  app_vm_size         = "Standard_D16s_v4"
  app_node_count      = 3
  app_min_count       = 3
  app_max_count       = 30
}

# AKS Module (Secondary)
module "aks_secondary" {
  source = "../../modules/aks"
  
  name_prefix = "revai-${local.environment}-secondary"
  location    = local.secondary_location
  tags        = local.common_tags
  
  kubernetes_version = "1.28"
  node_count         = 3
  vm_size           = "Standard_D8s_v4"
  enable_auto_scaling = true
  min_count         = 2
  max_count         = 15
  enable_monitoring = true
  log_analytics_workspace_id = azurerm_log_analytics_workspace.secondary.id
  
  vnet_address_space     = ["10.3.0.0/16"]
  subnet_address_prefixes = ["10.3.1.0/24"]
  pod_subnet_address_prefixes = ["10.3.2.0/24"]
  
  enable_app_node_pool = true
  app_vm_size         = "Standard_D16s_v4"
  app_node_count      = 2
  app_min_count       = 2
  app_max_count       = 20
}

# Storage Module (Primary)
module "storage_primary" {
  source = "../../modules/storage"
  
  name_prefix = "revai-${local.environment}-primary"
  location    = local.primary_location
  tags        = local.common_tags
  
  enable_postgresql = true
  postgresql_admin_username = "postgresadmin"
  postgresql_admin_password = var.postgresql_password
  postgresql_database_name = "revai_prod"
  postgresql_sku = "GP_Standard_D8s_v3"
  postgresql_storage_mb = 131072
  postgresql_backup_retention_days = 30
  postgresql_geo_redundant_backup = true
  postgresql_high_availability_mode = "ZoneRedundant"
  
  enable_redis = true
  redis_sku = "Premium"
  redis_capacity = 4
  redis_family = "P"
  
  enable_storage_account = true
  storage_account_tier = "Standard"
  storage_account_replication_type = "GRS"
}

# Storage Module (Secondary)
module "storage_secondary" {
  source = "../../modules/storage"
  
  name_prefix = "revai-${local.environment}-secondary"
  location    = local.secondary_location
  tags        = local.common_tags
  
  enable_postgresql = true
  postgresql_admin_username = "postgresadmin"
  postgresql_admin_password = var.postgresql_password
  postgresql_database_name = "revai_prod_dr"
  postgresql_sku = "GP_Standard_D4s_v3"
  postgresql_storage_mb = 65536
  postgresql_backup_retention_days = 30
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

# Security Module (Primary)
module "security_primary" {
  source = "../../modules/security"
  
  name_prefix = "revai-${local.environment}-primary"
  location    = local.primary_location
  tags        = local.common_tags
  
  enable_key_vault = true
  tenant_id = data.azurerm_client_config.current.tenant_id
  current_user_object_id = data.azurerm_client_config.current.object_id
  aks_managed_identity_id = module.aks_primary.aks_cluster_identity[0].principal_id
  
  enable_purge_protection = true
  soft_delete_retention_days = 90
  
  postgresql_password = var.postgresql_password
  redis_password = var.redis_password
  jwt_secret = var.jwt_secret
  
  enable_policy_assignment = true
  policy_scope = azurerm_resource_group.primary.id
  
  enable_security_center = true
}

# Security Module (Secondary)
module "security_secondary" {
  source = "../../modules/security"
  
  name_prefix = "revai-${local.environment}-secondary"
  location    = local.secondary_location
  tags        = local.common_tags
  
  enable_key_vault = true
  tenant_id = data.azurerm_client_config.current.tenant_id
  current_user_object_id = data.azurerm_client_config.current.object_id
  aks_managed_identity_id = module.aks_secondary.aks_cluster_identity[0].principal_id
  
  enable_purge_protection = true
  soft_delete_retention_days = 90
  
  postgresql_password = var.postgresql_password
  redis_password = var.redis_password
  jwt_secret = var.jwt_secret
  
  enable_policy_assignment = true
  policy_scope = azurerm_resource_group.secondary.id
  
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
output "primary_resource_group_name" {
  description = "Name of the primary resource group"
  value       = azurerm_resource_group.primary.name
}

output "secondary_resource_group_name" {
  description = "Name of the secondary resource group"
  value       = azurerm_resource_group.secondary.name
}

output "primary_aks_cluster_name" {
  description = "Name of the primary AKS cluster"
  value       = module.aks_primary.aks_cluster_name
}

output "secondary_aks_cluster_name" {
  description = "Name of the secondary AKS cluster"
  value       = module.aks_secondary.aks_cluster_name
}

output "primary_container_registry_name" {
  description = "Name of the primary container registry"
  value       = azurerm_container_registry.primary.name
}

output "secondary_container_registry_name" {
  description = "Name of the secondary container registry"
  value       = azurerm_container_registry.secondary.name
}

output "primary_key_vault_uri" {
  description = "URI of the primary Key Vault"
  value       = module.security_primary.key_vault_uri
}

output "secondary_key_vault_uri" {
  description = "URI of the secondary Key Vault"
  value       = module.security_secondary.key_vault_uri
}
