#!/bin/bash
# Production Deployment Scripts for RevAI Pro Platform
# Comprehensive deployment automation with health checks and rollback capabilities

set -euo pipefail

# Configuration
PROJECT_NAME="revai-pro"
NAMESPACE="revai-prod"
REGISTRY="your-registry.com"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
BACKUP_DIR="/opt/backups/revai-pro"
LOG_DIR="/var/log/revai-pro"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. Consider using a non-root user with sudo privileges."
    fi
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm" "jq" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "$tool is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi
    
    # Check Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check Helm repositories
    if ! helm repo list | grep -q "stable"; then
        log_info "Adding stable Helm repository..."
        helm repo add stable https://charts.helm.sh/stable
        helm repo update
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespaces and RBAC
setup_namespace() {
    log_info "Setting up namespace and RBAC..."
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create RBAC resources
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: revai-pro-sa
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: $NAMESPACE
  name: revai-pro-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: revai-pro-rolebinding
  namespace: $NAMESPACE
subjects:
- kind: ServiceAccount
  name: revai-pro-sa
  namespace: $NAMESPACE
roleRef:
  kind: Role
  name: revai-pro-role
  apiGroup: rbac.authorization.k8s.io
EOF
    
    log_success "Namespace and RBAC setup completed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local services=(
        "ai-orchestrator:8001"
        "agent-registry:8002"
        "routing-orchestrator:8003"
        "kpi-exporter:8004"
        "confidence-thresholds:8005"
        "model-audit:8006"
        "calendar-automation:8007"
        "letsmeet-automation:8008"
        "cruxx-automation:8009"
        "run-trace-schema:8010"
        "dlq-replay-tooling:8011"
        "metrics-exporter:8012"
        "event-bus-schema-registry:8013"
    )
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        log_info "Building image for $service_name..."
        
        # Build image
        docker build \
            -f Dockerfile.microservices \
            -t "$REGISTRY/$PROJECT_NAME-$service_name:$VERSION" \
            -t "$REGISTRY/$PROJECT_NAME-$service_name:latest" \
            --build-arg SERVICE_NAME="$service_name" \
            --build-arg SERVICE_PORT="$port" \
            .
        
        # Push image
        docker push "$REGISTRY/$PROJECT_NAME-$service_name:$VERSION"
        docker push "$REGISTRY/$PROJECT_NAME-$service_name:latest"
        
        log_success "Image built and pushed for $service_name"
    done
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL with HA
    helm upgrade --install postgresql stable/postgresql \
        --namespace "$NAMESPACE" \
        --set postgresqlPassword=revai_prod_password \
        --set postgresqlDatabase=revai_prod \
        --set postgresqlUsername=revai_user \
        --set replication.enabled=true \
        --set replication.slaveReplicas=2 \
        --set metrics.enabled=true \
        --set persistence.enabled=true \
        --set persistence.size=100Gi \
        --set resources.requests.memory=2Gi \
        --set resources.requests.cpu=1000m \
        --set resources.limits.memory=4Gi \
        --set resources.limits.cpu=2000m
    
    # Deploy Redis with HA
    helm upgrade --install redis stable/redis \
        --namespace "$NAMESPACE" \
        --set auth.enabled=true \
        --set auth.password=revai_redis_password \
        --set cluster.enabled=true \
        --set cluster.slaveCount=2 \
        --set metrics.enabled=true \
        --set persistence.enabled=true \
        --set persistence.size=50Gi \
        --set resources.requests.memory=1Gi \
        --set resources.requests.cpu=500m \
        --set resources.limits.memory=2Gi \
        --set resources.limits.cpu=1000m
    
    # Deploy Kafka
    helm upgrade --install kafka stable/kafka \
        --namespace "$NAMESPACE" \
        --set replicas=3 \
        --set persistence.enabled=true \
        --set persistence.size=100Gi \
        --set resources.requests.memory=2Gi \
        --set resources.requests.cpu=1000m \
        --set resources.limits.memory=4Gi \
        --set resources.limits.cpu=2000m
    
    # Deploy Prometheus for monitoring
    helm upgrade --install prometheus stable/prometheus \
        --namespace "$NAMESPACE" \
        --set server.persistentVolume.enabled=true \
        --set server.persistentVolume.size=50Gi \
        --set server.resources.requests.memory=2Gi \
        --set server.resources.requests.cpu=1000m \
        --set server.resources.limits.memory=4Gi \
        --set server.resources.limits.cpu=2000m
    
    # Deploy Grafana for visualization
    helm upgrade --install grafana stable/grafana \
        --namespace "$NAMESPACE" \
        --set persistence.enabled=true \
        --set persistence.size=10Gi \
        --set adminPassword=revai_grafana_password \
        --set resources.requests.memory=512Mi \
        --set resources.requests.cpu=250m \
        --set resources.limits.memory=1Gi \
        --set resources.limits.cpu=500m
    
    # Deploy ELK Stack for logging
    helm upgrade --install elasticsearch stable/elasticsearch \
        --namespace "$NAMESPACE" \
        --set replicas=3 \
        --set persistence.enabled=true \
        --set persistence.size=100Gi \
        --set resources.requests.memory=4Gi \
        --set resources.requests.cpu=2000m \
        --set resources.limits.memory=8Gi \
        --set resources.limits.cpu=4000m
    
    helm upgrade --install kibana stable/kibana \
        --namespace "$NAMESPACE" \
        --set resources.requests.memory=1Gi \
        --set resources.requests.cpu=500m \
        --set resources.limits.memory=2Gi \
        --set resources.limits.cpu=1000m
    
    helm upgrade --install logstash stable/logstash \
        --namespace "$NAMESPACE" \
        --set resources.requests.memory=1Gi \
        --set resources.requests.cpu=500m \
        --set resources.limits.memory=2Gi \
        --set resources.limits.cpu=1000m
    
    log_success "Infrastructure components deployed"
}

# Deploy application services
deploy_application() {
    log_info "Deploying application services..."
    
    # Create ConfigMaps
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: revai-pro-config
  namespace: $NAMESPACE
data:
  ENVIRONMENT: "$ENVIRONMENT"
  LOG_LEVEL: "INFO"
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
  DATABASE_URL: "postgresql://revai_user:revai_prod_password@postgresql:5432/revai_prod"
  REDIS_URL: "redis://:revai_redis_password@redis:6379"
  KAFKA_BROKERS: "kafka:9092"
EOF
    
    # Create Secrets
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: revai-pro-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  JWT_SECRET: $(echo -n "revai_jwt_secret_key" | base64)
  ENCRYPTION_KEY: $(echo -n "revai_encryption_key_32_chars" | base64)
  API_KEY: $(echo -n "revai_api_key" | base64)
EOF
    
    # Deploy services
    local services=(
        "ai-orchestrator:8001"
        "agent-registry:8002"
        "routing-orchestrator:8003"
        "kpi-exporter:8004"
        "confidence-thresholds:8005"
        "model-audit:8006"
        "calendar-automation:8007"
        "letsmeet-automation:8008"
        "cruxx-automation:8009"
        "run-trace-schema:8010"
        "dlq-replay-tooling:8011"
        "metrics-exporter:8012"
        "event-bus-schema-registry:8013"
    )
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        log_info "Deploying $service_name..."
        
        cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $service_name
  namespace: $NAMESPACE
  labels:
    app: $service_name
    version: $VERSION
spec:
  replicas: 3
  selector:
    matchLabels:
      app: $service_name
  template:
    metadata:
      labels:
        app: $service_name
        version: $VERSION
    spec:
      serviceAccountName: revai-pro-sa
      containers:
      - name: $service_name
        image: $REGISTRY/$PROJECT_NAME-$service_name:$VERSION
        ports:
        - containerPort: $port
        env:
        - name: SERVICE_NAME
          value: "$service_name"
        - name: SERVICE_PORT
          value: "$port"
        envFrom:
        - configMapRef:
            name: revai-pro-config
        - secretRef:
            name: revai-pro-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: $port
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: $port
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: $service_name
  namespace: $NAMESPACE
  labels:
    app: $service_name
spec:
  selector:
    app: $service_name
  ports:
  - port: $port
    targetPort: $port
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $service_name-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $service_name
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
        
        log_success "$service_name deployed"
    done
}

# Deploy ingress and load balancer
deploy_ingress() {
    log_info "Deploying ingress and load balancer..."
    
    # Deploy NGINX Ingress Controller
    helm upgrade --install nginx-ingress stable/nginx-ingress \
        --namespace "$NAMESPACE" \
        --set controller.service.type=LoadBalancer \
        --set controller.service.externalTrafficPolicy=Local \
        --set controller.resources.requests.memory=512Mi \
        --set controller.resources.requests.cpu=250m \
        --set controller.resources.limits.memory=1Gi \
        --set controller.resources.limits.cpu=500m
    
    # Create Ingress
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: revai-pro-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.revai-pro.com
    secretName: revai-pro-tls
  rules:
  - host: api.revai-pro.com
    http:
      paths:
      - path: /orchestrate
        pathType: Prefix
        backend:
          service:
            name: ai-orchestrator
            port:
              number: 8001
      - path: /agents
        pathType: Prefix
        backend:
          service:
            name: agent-registry
            port:
              number: 8002
      - path: /routing
        pathType: Prefix
        backend:
          service:
            name: routing-orchestrator
            port:
              number: 8003
      - path: /kpis
        pathType: Prefix
        backend:
          service:
            name: kpi-exporter
            port:
              number: 8004
      - path: /confidence
        pathType: Prefix
        backend:
          service:
            name: confidence-thresholds
            port:
              number: 8005
      - path: /audit
        pathType: Prefix
        backend:
          service:
            name: model-audit
            port:
              number: 8006
      - path: /calendar-automation
        pathType: Prefix
        backend:
          service:
            name: calendar-automation
            port:
              number: 8007
      - path: /letsmeet-automation
        pathType: Prefix
        backend:
          service:
            name: letsmeet-automation
            port:
              number: 8008
      - path: /cruxx-automation
        pathType: Prefix
        backend:
          service:
            name: cruxx-automation
            port:
              number: 8009
      - path: /traces
        pathType: Prefix
        backend:
          service:
            name: run-trace-schema
            port:
              number: 8010
      - path: /dlq
        pathType: Prefix
        backend:
          service:
            name: dlq-replay-tooling
            port:
              number: 8011
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: metrics-exporter
            port:
              number: 8012
      - path: /events
        pathType: Prefix
        backend:
          service:
            name: event-bus-schema-registry
            port:
              number: 8013
EOF
    
    log_success "Ingress and load balancer deployed"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    local services=(
        "ai-orchestrator:8001"
        "agent-registry:8002"
        "routing-orchestrator:8003"
        "kpi-exporter:8004"
        "confidence-thresholds:8005"
        "model-audit:8006"
        "calendar-automation:8007"
        "letsmeet-automation:8008"
        "cruxx-automation:8009"
        "run-trace-schema:8010"
        "dlq-replay-tooling:8011"
        "metrics-exporter:8012"
        "event-bus-schema-registry:8013"
    )
    
    local failed_services=()
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        log_info "Checking health of $service_name..."
        
        # Wait for service to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/"$service_name" -n "$NAMESPACE" || {
            log_error "$service_name is not ready"
            failed_services+=("$service_name")
            continue
        }
        
        # Check health endpoint
        local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$service_name" -o jsonpath='{.items[0].metadata.name}')
        
        if kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -f http://localhost:"$port"/health &> /dev/null; then
            log_success "$service_name is healthy"
        else
            log_error "$service_name health check failed"
            failed_services+=("$service_name")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_success "All services are healthy"
        return 0
    else
        log_error "Failed services: ${failed_services[*]}"
        return 1
    fi
}

# Database migration
run_database_migration() {
    log_info "Running database migration..."
    
    # Create migration job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: revai-pro-migration
  namespace: $NAMESPACE
spec:
  template:
    spec:
      serviceAccountName: revai-pro-sa
      containers:
      - name: migration
        image: $REGISTRY/$PROJECT_NAME-ai-orchestrator:$VERSION
        command: ["python", "-c", "import subprocess; subprocess.run(['psql', 'postgresql://revai_user:revai_prod_password@postgresql:5432/revai_prod', '-f', '/app/database/migrations/001_complete_schema_migration.sql'])"]
        envFrom:
        - configMapRef:
            name: revai-pro-config
        - secretRef:
            name: revai-pro-secrets
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete --timeout=600s job/revai-pro-migration -n "$NAMESPACE"
    
    if [ $? -eq 0 ]; then
        log_success "Database migration completed"
    else
        log_error "Database migration failed"
        return 1
    fi
}

# Create backup
create_backup() {
    log_info "Creating backup..."
    
    local backup_timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup database
    kubectl exec -n "$NAMESPACE" deployment/postgresql -- pg_dump -U revai_user revai_prod > "$backup_path/database.sql"
    
    # Backup configurations
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_path/k8s-resources.yaml"
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_path/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_path/secrets.yaml"
    
    # Backup persistent volumes
    kubectl get pv -o yaml > "$backup_path/persistent-volumes.yaml"
    
    # Compress backup
    tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "backup_$backup_timestamp"
    rm -rf "$backup_path"
    
    log_success "Backup created: $backup_path.tar.gz"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    local services=(
        "ai-orchestrator"
        "agent-registry"
        "routing-orchestrator"
        "kpi-exporter"
        "confidence-thresholds"
        "model-audit"
        "calendar-automation"
        "letsmeet-automation"
        "cruxx-automation"
        "run-trace-schema"
        "dlq-replay-tooling"
        "metrics-exporter"
        "event-bus-schema-registry"
    )
    
    for service in "${services[@]}"; do
        log_info "Rolling back $service..."
        kubectl rollout undo deployment/"$service" -n "$NAMESPACE"
        kubectl rollout status deployment/"$service" -n "$NAMESPACE"
    done
    
    log_success "Rollback completed"
}

# Main deployment function
deploy() {
    log_info "Starting production deployment..."
    
    check_prerequisites
    setup_namespace
    build_and_push_images
    deploy_infrastructure
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure to be ready..."
    sleep 60
    
    deploy_application
    deploy_ingress
    run_database_migration
    
    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    sleep 120
    
    if perform_health_checks; then
        log_success "Production deployment completed successfully!"
    else
        log_error "Health checks failed. Rolling back..."
        rollback_deployment
        error_exit "Deployment failed"
    fi
}

# Main script
main() {
    case "${1:-deploy}" in
        "deploy")
            deploy
            ;;
        "health-check")
            perform_health_checks
            ;;
        "backup")
            create_backup
            ;;
        "rollback")
            rollback_deployment
            ;;
        "migrate")
            run_database_migration
            ;;
        *)
            echo "Usage: $0 {deploy|health-check|backup|rollback|migrate}"
            exit 1
            ;;
    esac
}

main "$@"
