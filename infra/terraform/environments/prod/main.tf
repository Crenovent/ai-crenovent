# Simple test configuration for production environment
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
  environment = "prod"
  location    = "East US"

  common_tags = {
    Environment = local.environment
    Project     = "RevAI"
    ManagedBy   = "Terraform"
    Owner       = "devops"
    CostCenter  = "engineering"
    Compliance  = "SOC2-ISO27001"
  }
}

# Import existing resource group
import {
  to = azurerm_resource_group.main
  id = "/subscriptions/0a19726d-3c64-454b-b0d3-58f055e9d39a/resourceGroups/rg-revai-prod"
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
  retention_in_days   = 365
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
  sku                 = "Premium"
  admin_enabled       = true
  tags                = local.common_tags
}

# Service Bus
resource "azurerm_servicebus_namespace" "main" {
  name                         = "sb-revai-${local.environment}"
  location                     = azurerm_resource_group.main.location
  resource_group_name          = azurerm_resource_group.main.name
  sku                          = "Premium"
  premium_messaging_partitions = 4
  tags                         = local.common_tags
}

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