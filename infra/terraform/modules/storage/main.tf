# ================================================================================
# Storage Module - PostgreSQL, Redis, and Storage Accounts
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
resource "azurerm_resource_group" "storage" {
  name     = "${var.name_prefix}-storage-rg"
  location = var.location
  tags     = var.tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server" "main" {
  count               = var.enable_postgresql ? 1 : 0
  name                = "${var.name_prefix}-postgres"
  resource_group_name = azurerm_resource_group.storage.name
  location            = azurerm_resource_group.storage.location
  version             = "15"
  administrator_login = var.postgresql_admin_username
  administrator_password = var.postgresql_admin_password
  zone                = "1"
  tags                = var.tags

  storage_mb = var.postgresql_storage_mb
  sku_name   = var.postgresql_sku

  backup_retention_days        = var.postgresql_backup_retention_days
  geo_redundant_backup_enabled = var.postgresql_geo_redundant_backup

  high_availability {
    mode = var.postgresql_high_availability_mode
  }

  maintenance_window {
    day_of_week  = 0
    start_hour   = 8
    start_minute = 0
  }

  depends_on = [azurerm_resource_group.storage]
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "main" {
  count     = var.enable_postgresql ? 1 : 0
  name      = var.postgresql_database_name
  server_id = azurerm_postgresql_flexible_server.main[0].id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  count               = var.enable_redis ? 1 : 0
  name                = "${var.name_prefix}-redis"
  location            = azurerm_resource_group.storage.location
  resource_group_name = azurerm_resource_group.storage.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  tags                = var.tags

  redis_configuration {
    maxmemory_reserved = 2
    maxmemory_delta    = 2
    maxmemory_policy   = "allkeys-lru"
  }

  depends_on = [azurerm_resource_group.storage]
}

# Storage Account for Blob Storage
resource "azurerm_storage_account" "main" {
  count                    = var.enable_storage_account ? 1 : 0
  name                     = "${var.name_prefix}storage"
  resource_group_name      = azurerm_resource_group.storage.name
  location                 = azurerm_resource_group.storage.location
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_account_replication_type
  account_kind             = "StorageV2"
  tags                     = var.tags

  blob_properties {
    versioning_enabled  = true
    change_feed_enabled = true
    change_feed_retention_in_days = 7
  }

  depends_on = [azurerm_resource_group.storage]
}

# Storage Container for Application Data
resource "azurerm_storage_container" "app_data" {
  count                 = var.enable_storage_account ? 1 : 0
  name                  = "app-data"
  storage_account_name  = azurerm_storage_account.main[0].name
  container_access_type = "private"
}

# Storage Container for Logs
resource "azurerm_storage_container" "logs" {
  count                 = var.enable_storage_account ? 1 : 0
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.main[0].name
  container_access_type = "private"
}

# Storage Container for Backups
resource "azurerm_storage_container" "backups" {
  count                 = var.enable_storage_account ? 1 : 0
  name                  = "backups"
  storage_account_name  = azurerm_storage_account.main[0].name
  container_access_type = "private"
}

# Outputs
output "postgresql_server_id" {
  description = "The ID of the PostgreSQL server"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server.main[0].id : null
}

output "postgresql_server_fqdn" {
  description = "The FQDN of the PostgreSQL server"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server.main[0].fqdn : null
}

output "postgresql_database_name" {
  description = "The name of the PostgreSQL database"
  value       = var.enable_postgresql ? azurerm_postgresql_flexible_server_database.main[0].name : null
}

output "redis_cache_id" {
  description = "The ID of the Redis cache"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].id : null
}

output "redis_cache_hostname" {
  description = "The hostname of the Redis cache"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].hostname : null
}

output "redis_cache_port" {
  description = "The port of the Redis cache"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].port : null
}

output "storage_account_id" {
  description = "The ID of the storage account"
  value       = var.enable_storage_account ? azurerm_storage_account.main[0].id : null
}

output "storage_account_name" {
  description = "The name of the storage account"
  value       = var.enable_storage_account ? azurerm_storage_account.main[0].name : null
}

output "storage_account_primary_endpoint" {
  description = "The primary endpoint of the storage account"
  value       = var.enable_storage_account ? azurerm_storage_account.main[0].primary_blob_endpoint : null
}

output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.storage.name
}
