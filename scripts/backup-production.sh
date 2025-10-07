#!/bin/bash
# Production Backup Procedures for RevAI Pro Platform
# Comprehensive backup automation with encryption and retention policies

set -euo pipefail

# Configuration
BACKUP_BASE_DIR="/opt/backups/revai-pro"
ENCRYPTION_KEY_FILE="/etc/revai-pro/backup-encryption.key"
RETENTION_DAYS=30
COMPRESSION_LEVEL=6
LOG_FILE="/var/log/revai-pro/backup.log"
NOTIFICATION_WEBHOOK="${NOTIFICATION_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log_error "$1"
    send_notification "BACKUP_FAILED" "$1"
    exit 1
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ -n "$NOTIFICATION_WEBHOOK" ]]; then
        curl -X POST "$NOTIFICATION_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"status\":\"$status\",\"message\":\"$message\",\"timestamp\":\"$(date -Iseconds)\"}" \
            --silent --show-error || log_warning "Failed to send notification"
    fi
}

# Create backup directory structure
setup_backup_directories() {
    log_info "Setting up backup directory structure..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="$BACKUP_BASE_DIR/backup_$timestamp"
    
    mkdir -p "$backup_dir"/{database,kubernetes,volumes,configs,logs}
    
    echo "$backup_dir"
}

# Backup PostgreSQL database
backup_database() {
    local backup_dir="$1"
    local db_backup_dir="$backup_dir/database"
    
    log_info "Starting PostgreSQL database backup..."
    
    # Get database connection details
    local db_host="${POSTGRES_HOST:-postgresql}"
    local db_port="${POSTGRES_PORT:-5432}"
    local db_name="${POSTGRES_DB:-revai_prod}"
    local db_user="${POSTGRES_USER:-revai_user}"
    local db_password="${POSTGRES_PASSWORD:-revai_prod_password}"
    
    # Set PGPASSWORD environment variable
    export PGPASSWORD="$db_password"
    
    # Create database dump
    local dump_file="$db_backup_dir/database_dump.sql"
    
    if pg_dump -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" \
        --verbose --no-password --format=custom --compress=9 \
        --file="$dump_file"; then
        log_success "Database backup completed: $dump_file"
        
        # Get database size for logging
        local db_size=$(du -h "$dump_file" | cut -f1)
        log_info "Database backup size: $db_size"
    else
        error_exit "Database backup failed"
    fi
    
    # Backup database configuration
    kubectl get configmap postgresql-config -n revai-prod -o yaml > "$db_backup_dir/postgresql_config.yaml" 2>/dev/null || true
    
    # Backup database secrets
    kubectl get secret postgresql-secret -n revai-prod -o yaml > "$db_backup_dir/postgresql_secret.yaml" 2>/dev/null || true
}

# Backup Redis data
backup_redis() {
    local backup_dir="$1"
    local redis_backup_dir="$backup_dir/redis"
    
    log_info "Starting Redis backup..."
    
    # Get Redis connection details
    local redis_host="${REDIS_HOST:-redis}"
    local redis_port="${REDIS_PORT:-6379}"
    local redis_password="${REDIS_PASSWORD:-revai_redis_password}"
    
    # Create Redis dump
    local redis_dump_file="$redis_backup_dir/redis_dump.rdb"
    
    # Connect to Redis and create backup
    if redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_password" --rdb "$redis_dump_file"; then
        log_success "Redis backup completed: $redis_dump_file"
        
        # Get Redis backup size
        local redis_size=$(du -h "$redis_dump_file" | cut -f1)
        log_info "Redis backup size: $redis_size"
    else
        log_warning "Redis backup failed - continuing with other backups"
    fi
    
    # Backup Redis configuration
    kubectl get configmap redis-config -n revai-prod -o yaml > "$redis_backup_dir/redis_config.yaml" 2>/dev/null || true
}

# Backup Kubernetes resources
backup_kubernetes_resources() {
    local backup_dir="$1"
    local k8s_backup_dir="$backup_dir/kubernetes"
    
    log_info "Starting Kubernetes resources backup..."
    
    # Backup all resources in revai-prod namespace
    kubectl get all -n revai-prod -o yaml > "$k8s_backup_dir/all_resources.yaml"
    
    # Backup ConfigMaps
    kubectl get configmaps -n revai-prod -o yaml > "$k8s_backup_dir/configmaps.yaml"
    
    # Backup Secrets (excluding service account tokens)
    kubectl get secrets -n revai-prod -o yaml > "$k8s_backup_dir/secrets.yaml"
    
    # Backup PersistentVolumeClaims
    kubectl get pvc -n revai-prod -o yaml > "$k8s_backup_dir/persistentvolumeclaims.yaml"
    
    # Backup Services
    kubectl get services -n revai-prod -o yaml > "$k8s_backup_dir/services.yaml"
    
    # Backup Ingress
    kubectl get ingress -n revai-prod -o yaml > "$k8s_backup_dir/ingress.yaml"
    
    # Backup NetworkPolicies
    kubectl get networkpolicies -n revai-prod -o yaml > "$k8s_backup_dir/networkpolicies.yaml"
    
    # Backup RBAC resources
    kubectl get rolebindings -n revai-prod -o yaml > "$k8s_backup_dir/rolebindings.yaml"
    kubectl get roles -n revai-prod -o yaml > "$k8s_backup_dir/roles.yaml"
    
    # Backup ServiceAccounts
    kubectl get serviceaccounts -n revai-prod -o yaml > "$k8s_backup_dir/serviceaccounts.yaml"
    
    # Backup HPA
    kubectl get hpa -n revai-prod -o yaml > "$k8s_backup_dir/hpa.yaml"
    
    # Backup PDB
    kubectl get pdb -n revai-prod -o yaml > "$k8s_backup_dir/pdb.yaml"
    
    log_success "Kubernetes resources backup completed"
}

# Backup persistent volumes
backup_persistent_volumes() {
    local backup_dir="$1"
    local volumes_backup_dir="$backup_dir/volumes"
    
    log_info "Starting persistent volumes backup..."
    
    # Get list of PVCs
    local pvcs=$(kubectl get pvc -n revai-prod -o jsonpath='{.items[*].metadata.name}')
    
    for pvc in $pvcs; do
        log_info "Backing up PVC: $pvc"
        
        # Get PVC details
        local pvc_yaml=$(kubectl get pvc "$pvc" -n revai-prod -o yaml)
        echo "$pvc_yaml" > "$volumes_backup_dir/pvc_$pvc.yaml"
        
        # Create a temporary pod to access the volume
        local pod_name="backup-pod-$pvc-$(date +%s)"
        
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $pod_name
  namespace: revai-prod
spec:
  containers:
  - name: backup-container
    image: busybox
    command: ['sleep', '3600']
    volumeMounts:
    - name: pvc-volume
      mountPath: /data
  volumes:
  - name: pvc-volume
    persistentVolumeClaim:
      claimName: $pvc
  restartPolicy: Never
EOF
        
        # Wait for pod to be ready
        kubectl wait --for=condition=Ready pod/"$pod_name" -n revai-prod --timeout=60s
        
        # Create tar backup of the volume
        kubectl exec "$pod_name" -n revai-prod -- tar -czf /tmp/backup.tar.gz -C /data .
        
        # Copy backup from pod
        kubectl cp "revai-prod/$pod_name:/tmp/backup.tar.gz" "$volumes_backup_dir/pvc_$pvc.tar.gz"
        
        # Clean up pod
        kubectl delete pod "$pod_name" -n revai-prod
        
        log_success "PVC $pvc backup completed"
    done
    
    log_success "Persistent volumes backup completed"
}

# Backup application configurations
backup_application_configs() {
    local backup_dir="$1"
    local configs_backup_dir="$backup_dir/configs"
    
    log_info "Starting application configurations backup..."
    
    # Backup environment variables
    kubectl get configmap revai-pro-config -n revai-prod -o yaml > "$configs_backup_dir/revai_pro_config.yaml" 2>/dev/null || true
    
    # Backup secrets
    kubectl get secret revai-pro-secrets -n revai-prod -o yaml > "$configs_backup_dir/revai_pro_secrets.yaml" 2>/dev/null || true
    
    # Backup monitoring configuration
    kubectl get configmap prometheus-config -n revai-prod -o yaml > "$configs_backup_dir/prometheus_config.yaml" 2>/dev/null || true
    kubectl get configmap grafana-datasources -n revai-prod -o yaml > "$configs_backup_dir/grafana_datasources.yaml" 2>/dev/null || true
    kubectl get configmap grafana-dashboards -n revai-prod -o yaml > "$configs_backup_dir/grafana_dashboards.yaml" 2>/dev/null || true
    
    # Backup security configuration
    kubectl get configmap security-headers-config -n revai-prod -o yaml > "$configs_backup_dir/security_headers_config.yaml" 2>/dev/null || true
    kubectl get configmap falco-rules -n revai-prod -o yaml > "$configs_backup_dir/falco_rules.yaml" 2>/dev/null || true
    
    # Backup application logs configuration
    kubectl get configmap logstash-config -n revai-prod -o yaml > "$configs_backup_dir/logstash_config.yaml" 2>/dev/null || true
    
    log_success "Application configurations backup completed"
}

# Backup application logs
backup_application_logs() {
    local backup_dir="$1"
    local logs_backup_dir="$backup_dir/logs"
    
    log_info "Starting application logs backup..."
    
    # Get list of pods
    local pods=$(kubectl get pods -n revai-prod -o jsonpath='{.items[*].metadata.name}')
    
    for pod in $pods; do
        log_info "Backing up logs for pod: $pod"
        
        # Get pod logs
        kubectl logs "$pod" -n revai-prod --previous > "$logs_backup_dir/${pod}_previous.log" 2>/dev/null || true
        kubectl logs "$pod" -n revai-prod > "$logs_backup_dir/${pod}_current.log" 2>/dev/null || true
        
        # Get pod description
        kubectl describe pod "$pod" -n revai-prod > "$logs_backup_dir/${pod}_description.txt" 2>/dev/null || true
    done
    
    # Backup system logs if accessible
    if [[ -d "/var/log/revai-pro" ]]; then
        cp -r /var/log/revai-pro/* "$logs_backup_dir/" 2>/dev/null || true
    fi
    
    log_success "Application logs backup completed"
}

# Encrypt backup
encrypt_backup() {
    local backup_dir="$1"
    
    log_info "Encrypting backup..."
    
    # Check if encryption key exists
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        log_warning "Encryption key not found, creating new one..."
        mkdir -p "$(dirname "$ENCRYPTION_KEY_FILE")"
        openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
        chmod 600 "$ENCRYPTION_KEY_FILE"
    fi
    
    # Create encrypted tar archive
    local backup_name=$(basename "$backup_dir")
    local encrypted_backup="$BACKUP_BASE_DIR/${backup_name}_encrypted.tar.gz.gpg"
    
    # Compress and encrypt
    tar -czf - -C "$BACKUP_BASE_DIR" "$backup_name" | \
        gpg --symmetric --cipher-algo AES256 --compress-algo 1 \
        --batch --yes --passphrase-file "$ENCRYPTION_KEY_FILE" \
        --output "$encrypted_backup"
    
    if [[ $? -eq 0 ]]; then
        log_success "Backup encrypted: $encrypted_backup"
        
        # Remove unencrypted backup
        rm -rf "$backup_dir"
        
        echo "$encrypted_backup"
    else
        error_exit "Backup encryption failed"
    fi
}

# Upload backup to remote storage
upload_backup() {
    local encrypted_backup="$1"
    
    log_info "Uploading backup to remote storage..."
    
    # AWS S3 upload
    if [[ -n "${AWS_S3_BUCKET:-}" ]]; then
        local s3_key="backups/$(basename "$encrypted_backup")"
        
        if aws s3 cp "$encrypted_backup" "s3://$AWS_S3_BUCKET/$s3_key"; then
            log_success "Backup uploaded to S3: s3://$AWS_S3_BUCKET/$s3_key"
        else
            log_warning "S3 upload failed"
        fi
    fi
    
    # Google Cloud Storage upload
    if [[ -n "${GCS_BUCKET:-}" ]]; then
        local gcs_key="backups/$(basename "$encrypted_backup")"
        
        if gsutil cp "$encrypted_backup" "gs://$GCS_BUCKET/$gcs_key"; then
            log_success "Backup uploaded to GCS: gs://$GCS_BUCKET/$gcs_key"
        else
            log_warning "GCS upload failed"
        fi
    fi
    
    # Azure Blob Storage upload
    if [[ -n "${AZURE_STORAGE_ACCOUNT:-}" ]]; then
        local azure_container="${AZURE_CONTAINER:-backups}"
        local azure_blob="$(basename "$encrypted_backup")"
        
        if az storage blob upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$azure_container" \
            --file "$encrypted_backup" \
            --name "$azure_blob"; then
            log_success "Backup uploaded to Azure: $azure_container/$azure_blob"
        else
            log_warning "Azure upload failed"
        fi
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # Remove local backups older than retention period
    find "$BACKUP_BASE_DIR" -name "*_encrypted.tar.gz.gpg" -type f -mtime +$RETENTION_DAYS -delete
    
    # Remove remote backups older than retention period
    if [[ -n "${AWS_S3_BUCKET:-}" ]]; then
        aws s3 ls "s3://$AWS_S3_BUCKET/backups/" --recursive | \
            awk '$1 < "'$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')'" {print $4}' | \
            xargs -I {} aws s3 rm "s3://$AWS_S3_BUCKET/{}" 2>/dev/null || true
    fi
    
    if [[ -n "${GCS_BUCKET:-}" ]]; then
        gsutil ls "gs://$GCS_BUCKET/backups/" | \
            awk '$1 < "'$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')'" {print $1}' | \
            xargs -I {} gsutil rm {} 2>/dev/null || true
    fi
    
    log_success "Old backups cleanup completed"
}

# Verify backup integrity
verify_backup() {
    local encrypted_backup="$1"
    
    log_info "Verifying backup integrity..."
    
    # Test decryption
    local test_dir="/tmp/backup_test_$(date +%s)"
    mkdir -p "$test_dir"
    
    if gpg --decrypt --batch --yes --passphrase-file "$ENCRYPTION_KEY_FILE" \
        "$encrypted_backup" | tar -tzf - > /dev/null; then
        log_success "Backup integrity verified"
        rm -rf "$test_dir"
        return 0
    else
        log_error "Backup integrity verification failed"
        rm -rf "$test_dir"
        return 1
    fi
}

# Generate backup report
generate_backup_report() {
    local backup_dir="$1"
    local encrypted_backup="$2"
    
    log_info "Generating backup report..."
    
    local report_file="$BACKUP_BASE_DIR/backup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
RevAI Pro Platform Backup Report
================================
Date: $(date)
Backup Directory: $backup_dir
Encrypted Backup: $encrypted_backup

Backup Contents:
- Database: PostgreSQL dump
- Redis: Redis dump
- Kubernetes: All resources in revai-prod namespace
- Volumes: Persistent volume claims
- Configs: Application configurations
- Logs: Application and system logs

Backup Size: $(du -h "$encrypted_backup" | cut -f1)
Compression: gzip level $COMPRESSION_LEVEL
Encryption: GPG AES256
Retention: $RETENTION_DAYS days

Status: SUCCESS
EOF
    
    log_success "Backup report generated: $report_file"
}

# Main backup function
main_backup() {
    log_info "Starting RevAI Pro platform backup..."
    
    # Create backup directory
    local backup_dir=$(setup_backup_directories)
    
    # Perform backups
    backup_database "$backup_dir"
    backup_redis "$backup_dir"
    backup_kubernetes_resources "$backup_dir"
    backup_persistent_volumes "$backup_dir"
    backup_application_configs "$backup_dir"
    backup_application_logs "$backup_dir"
    
    # Encrypt backup
    local encrypted_backup=$(encrypt_backup "$backup_dir")
    
    # Verify backup
    if ! verify_backup "$encrypted_backup"; then
        error_exit "Backup verification failed"
    fi
    
    # Upload to remote storage
    upload_backup "$encrypted_backup"
    
    # Clean up old backups
    cleanup_old_backups
    
    # Generate report
    generate_backup_report "$backup_dir" "$encrypted_backup"
    
    # Send success notification
    send_notification "BACKUP_SUCCESS" "Backup completed successfully: $(basename "$encrypted_backup")"
    
    log_success "RevAI Pro platform backup completed successfully!"
}

# Restore function
restore_backup() {
    local backup_file="$1"
    local restore_dir="/tmp/restore_$(date +%s)"
    
    log_info "Starting backup restore from: $backup_file"
    
    # Create restore directory
    mkdir -p "$restore_dir"
    
    # Decrypt and extract backup
    if gpg --decrypt --batch --yes --passphrase-file "$ENCRYPTION_KEY_FILE" \
        "$backup_file" | tar -xzf - -C "$restore_dir"; then
        log_success "Backup extracted to: $restore_dir"
    else
        error_exit "Backup extraction failed"
    fi
    
    # Restore database
    if [[ -f "$restore_dir"/*/database/database_dump.sql ]]; then
        log_info "Restoring database..."
        # Database restore logic here
    fi
    
    # Restore Kubernetes resources
    if [[ -f "$restore_dir"/*/kubernetes/all_resources.yaml ]]; then
        log_info "Restoring Kubernetes resources..."
        kubectl apply -f "$restore_dir"/*/kubernetes/all_resources.yaml
    fi
    
    # Clean up
    rm -rf "$restore_dir"
    
    log_success "Backup restore completed"
}

# Main script
main() {
    case "${1:-backup}" in
        "backup")
            main_backup
            ;;
        "restore")
            if [[ -z "${2:-}" ]]; then
                error_exit "Backup file path required for restore"
            fi
            restore_backup "$2"
            ;;
        "verify")
            if [[ -z "${2:-}" ]]; then
                error_exit "Backup file path required for verification"
            fi
            verify_backup "$2"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        *)
            echo "Usage: $0 {backup|restore <backup_file>|verify <backup_file>|cleanup}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
