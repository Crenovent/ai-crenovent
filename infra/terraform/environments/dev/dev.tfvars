# Development Environment Variables
environment = "dev"
location    = "Central India"

# PostgreSQL Configuration
postgresql_password      = "DevPostgres123!"
postgresql_database_name = "revai_dev"

# Redis Configuration  
redis_password = "DevRedis123!"

# JWT Configuration
jwt_secret = "dev-jwt-secret-key-change-in-production"

# AKS Configuration
aks_node_count = 2
aks_vm_size    = "Standard_D2s_v4"
aks_min_count  = 1
aks_max_count  = 5

# Storage Configuration
postgresql_sku        = "GP_Standard_D2s_v3"
postgresql_storage_mb = 32768
redis_capacity        = 1

# Security Configuration
enable_purge_protection    = false
soft_delete_retention_days = 7
enable_policy_assignment   = false
enable_security_center     = false

# Monitoring Configuration
log_analytics_retention_days = 30
enable_monitoring            = true

# Networking Configuration
vnet_address_space          = ["10.0.0.0/16"]
subnet_address_prefixes     = ["10.0.1.0/24"]
pod_subnet_address_prefixes = ["10.0.2.0/24"]

# Feature Flags
enable_cosmosdb = false
enable_openai   = false
enable_cdn      = false
enable_waf      = false
enable_backup   = true
