# Staging Environment Variables
environment = "staging"
location = "West Europe"

# PostgreSQL Configuration
postgresql_password = "StagingPostgres456!"
postgresql_database_name = "revai_staging"

# Redis Configuration  
redis_password = "StagingRedis456!"

# JWT Configuration
jwt_secret = "staging-jwt-secret-key-change-in-production"

# AKS Configuration
aks_node_count = 3
aks_vm_size = "Standard_D4s_v4"
aks_min_count = 2
aks_max_count = 8

# Storage Configuration
postgresql_sku = "GP_Standard_D4s_v3"
postgresql_storage_mb = 65536
redis_capacity = 2

# Security Configuration
enable_purge_protection = true
soft_delete_retention_days = 30
enable_policy_assignment = true
enable_security_center = true

# Monitoring Configuration
log_analytics_retention_days = 90
enable_monitoring = true

# Networking Configuration
vnet_address_space = ["10.1.0.0/16"]
subnet_address_prefixes = ["10.1.1.0/24"]
pod_subnet_address_prefixes = ["10.1.2.0/24"]

# Feature Flags
enable_cosmosdb = false
enable_openai = false
enable_cdn = true
enable_waf = true
enable_backup = true
