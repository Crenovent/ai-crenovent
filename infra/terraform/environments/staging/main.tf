# Simple test configuration for staging environment
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
  subscription_id = "0a19726d-3c64-454b-b0d3-58f055e9d39a"
  features {}
}

# Local variables
locals {
  environment = "staging"
  location    = "West Europe"

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

# Key Vault Module - Secrets Manager & PKI
module "keyvault" {
  source = "../../modules/keyvault"

  name_prefix         = "kv-revai-${local.environment}"
  location            = local.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  tags                = local.common_tags

  # Security Configuration
  enable_purge_protection    = true
  soft_delete_retention_days = 30
  network_default_action     = "Deny"

  # RBAC Configuration
  admin_object_ids = [
    data.azurerm_client_config.current.object_id
  ]

  # PKI Configuration
  enable_pki  = true
  pki_subject = "CN=RevAI Staging Root CA, O=Crenovent, C=US"
  pki_dns_names = [
    "staging.revai.crenovent.com",
    "*.staging.revai.crenovent.com"
  ]

  # Application Secrets
  application_secrets = {
    "postgresql-password" = "StagingPostgres123!"
    "redis-password"      = "StagingRedis123!"
    "jwt-secret"          = "staging-jwt-secret-key"
    "api-key"             = "staging-api-key-12345"
  }

  # Integration
  enable_aks_integration     = true
  enable_diagnostics         = true
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

# Data sources
data "azurerm_client_config" "current" {}

# Outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "container_registry_name" {
  description = "Name of the container registry"
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Login server of the container registry"
  value       = azurerm_container_registry.main.login_server
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = module.keyvault.key_vault_uri
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = module.keyvault.key_vault_name
}

output "evidence_signing_key_id" {
  description = "ID of the evidence signing key"
  value       = module.keyvault.evidence_signing_key_id
}

output "root_ca_certificate_id" {
  description = "ID of the root CA certificate"
  value       = module.keyvault.root_ca_certificate_id
}
