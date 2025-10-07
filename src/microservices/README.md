# AI Microservices Suite

This directory contains the AI microservices suite for RevAI Pro, implementing the automation layer with FastAPI + gRPC stubs.

## 🏗️ Architecture

The AI microservices suite consists of 6 core services:

### 1. **AI Orchestrator** (Port 8001)
- **Purpose**: Main orchestration service that coordinates all AI operations
- **Features**: Service discovery, health monitoring, request routing
- **API**: `/orchestrate` - Main orchestration endpoint

### 2. **Agent Registry** (Port 8002)
- **Purpose**: Manages AI agents registration, lifecycle, and configuration
- **Features**: Agent registration, querying, retirement, capability management
- **API**: `/agents/*` - Agent management endpoints

### 3. **Routing Orchestrator** (Port 8003)
- **Purpose**: Handles mode selection based on policy and confidence
- **Features**: Policy rules, confidence evaluation, UI mode routing
- **API**: `/routing/*` - Routing and policy endpoints

### 4. **KPI Exporter** (Port 8004)
- **Purpose**: Tracks and exports key performance indicators
- **Features**: Prompt conversion, override rate, trust drift metrics
- **API**: `/kpis/*` - KPI tracking and export endpoints

### 5. **Confidence Thresholds** (Port 8005)
- **Purpose**: Manages confidence thresholds and auto-fallback logic
- **Features**: Threshold configuration, explainability, fallback modes
- **API**: `/confidence/*` - Confidence evaluation endpoints

### 6. **Model Audit** (Port 8006)
- **Purpose**: Tracks and audits AI model calls, tools, and decisions
- **Features**: Call logging, tool usage tracking, decision audit trails
- **API**: `/audit/*` - Audit and tracing endpoints

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip3
- Docker (optional)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.microservices.txt
   ```

2. **Start all services**:
   ```bash
   # Linux/Mac
   ./start-microservices.sh
   
   # Windows
   start-microservices.bat
   ```

3. **Verify services are running**:
   ```bash
   curl http://localhost:8001/health
   ```

### Docker Deployment

1. **Build and start with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.microservices.yml up -d
   ```

2. **Check service status**:
   ```bash
   docker-compose -f docker-compose.microservices.yml ps
   ```

## 📊 Service Endpoints

| Service | Port | Health Check | API Docs |
|---------|------|--------------|----------|
| AI Orchestrator | 8001 | `/health` | `/docs` |
| Agent Registry | 8002 | `/health` | `/docs` |
| Routing Orchestrator | 8003 | `/health` | `/docs` |
| KPI Exporter | 8004 | `/health` | `/docs` |
| Confidence Thresholds | 8005 | `/health` | `/docs` |
| Model Audit | 8006 | `/health` | `/docs` |

## 🔧 Configuration

### Environment Variables

Each service can be configured with the following environment variables:

```bash
# Service identification
SERVICE_NAME=ai-orchestrator
SERVICE_PORT=8001

# Database (if using persistent storage)
DATABASE_URL=postgresql://user:pass@localhost:5432/revai_ai
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Service Configuration

Each service supports configuration through:

1. **Environment variables** (highest priority)
2. **Configuration files** (service-specific)
3. **Default values** (lowest priority)

## 📈 Monitoring & Observability

### Health Checks

All services expose health check endpoints at `/health`:

```bash
curl http://localhost:8001/health
```

### Metrics

Services expose Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8001/metrics
```

### Logging

Logs are written to:
- **Console**: Structured JSON logs
- **Files**: `logs/{service-name}.log`
- **Docker**: Container logs

## 🔄 API Usage Examples

### Orchestrate a Request

```bash
curl -X POST http://localhost:8001/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant-123",
    "user_id": "user-456",
    "session_id": "session-789",
    "service_name": "calendar",
    "operation_type": "create_event",
    "input_data": {
      "title": "Team Meeting",
      "start_time": "2024-01-15T10:00:00Z"
    },
    "context": {
      "complexity": "medium",
      "urgency": "high"
    }
  }'
```

### Register an Agent

```bash
curl -X POST http://localhost:8002/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Calendar Assistant",
    "description": "AI agent for calendar management",
    "capabilities": ["calendar_management"],
    "confidence_threshold": 0.8,
    "trust_score": 0.7,
    "tenant_id": "tenant-123",
    "region": "US"
  }'
```

### Query Agents

```bash
curl -X POST http://localhost:8002/agents/query \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant-123",
    "capabilities": ["calendar_management"],
    "status": "active",
    "min_trust_score": 0.7
  }'
```

## 🧪 Testing

### Unit Tests

```bash
pytest tests/microservices/ -v
```

### Integration Tests

```bash
pytest tests/integration/ -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8001
```

## 🔒 Security

### Authentication

Services support JWT-based authentication:

```bash
curl -H "Authorization: Bearer <jwt-token>" http://localhost:8001/orchestrate
```

### Rate Limiting

All services implement rate limiting:
- **Default**: 100 requests/minute per IP
- **Configurable**: Per tenant/user limits

### Data Residency

Services respect data residency requirements:
- **US**: Data stays in US regions
- **EU**: Data stays in EU regions
- **APAC**: Data stays in APAC regions

## 🚨 Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   # Check logs
   tail -f logs/{service-name}.log
   
   # Check port availability
   netstat -tulpn | grep :8001
   ```

2. **Service unhealthy**:
   ```bash
   # Check health endpoint
   curl http://localhost:8001/health
   
   # Check dependencies
   curl http://localhost:8002/health
   ```

3. **High memory usage**:
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check for memory leaks
   python -m memory_profiler src/microservices/orchestrator.py
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python src/microservices/orchestrator.py
```

## 📚 Development

### Adding a New Service

1. **Create service file**:
   ```python
   # src/microservices/new_service.py
   from fastapi import FastAPI
   
   app = FastAPI(title="New Service")
   
   @app.get("/health")
   async def health_check():
       return {"status": "healthy"}
   ```

2. **Add to Docker Compose**:
   ```yaml
   new-service:
     build:
       context: .
       dockerfile: Dockerfile.microservices
     ports:
       - "8007:8007"
     environment:
       - SERVICE_NAME=new-service
       - SERVICE_PORT=8007
   ```

3. **Update orchestrator**:
   ```python
   SERVICE_ENDPOINTS["new_service"] = "http://localhost:8007"
   ```

### Code Style

- **Formatting**: Black
- **Linting**: Flake8
- **Type Checking**: MyPy
- **Testing**: Pytest

```bash
# Format code
black src/microservices/

# Lint code
flake8 src/microservices/

# Type check
mypy src/microservices/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- **Documentation**: [API Docs](http://localhost:8001/docs)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
