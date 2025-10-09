# Production Environment Variables
environment = "prod"
primary_location = "Central India"
secondary_location = "West Europe"

# PostgreSQL Configuration
postgresql_password = "ProdPostgres789!"
postgresql_database_name = "revai_prod"

# Redis Configuration  
redis_password = "ProdRedis789!"

# JWT Configuration
jwt_secret = "prod-jwt-secret-key-change-in-production"

# AKS Configuration (Primary)
aks_primary_node_count = 5
aks_primary_vm_size = "Standard_D8s_v4"
aks_primary_min_count = 3
aks_primary_max_count = 20

# AKS Configuration (Secondary)
aks_secondary_node_count = 3
aks_secondary_vm_size = "Standard_D8s_v4"
aks_secondary_min_count = 2
aks_secondary_max_count = 15

# Storage Configuration (Primary)
postgresql_primary_sku = "GP_Standard_D8s_v3"
postgresql_primary_storage_mb = 131072
redis_primary_capacity = 4

# Storage Configuration (Secondary)
postgresql_secondary_sku = "GP_Standard_D4s_v3"
postgresql_secondary_storage_mb = 65536
redis_secondary_capacity = 2

# Security Configuration
enable_purge_protection = true
soft_delete_retention_days = 90
enable_policy_assignment = true
enable_security_center = true

# Monitoring Configuration
log_analytics_retention_days = 2555  # 7 years for compliance
enable_monitoring = true

# Networking Configuration (Primary)
primary_vnet_address_space = ["10.2.0.0/16"]
primary_subnet_address_prefixes = ["10.2.1.0/24"]
primary_pod_subnet_address_prefixes = ["10.2.2.0/24"]

# Networking Configuration (Secondary)
secondary_vnet_address_space = ["10.3.0.0/16"]
secondary_subnet_address_prefixes = ["10.3.1.0/24"]
secondary_pod_subnet_address_prefixes = ["10.3.2.0/24"]

# Feature Flags
enable_cosmosdb = true
enable_openai = true
enable_cdn = true
enable_waf = true
enable_backup = true
enable_disaster_recovery = true
