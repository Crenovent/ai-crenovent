#!/bin/bash

# AI Microservices Startup Script
# Starts all AI microservices with proper configuration

set -e

echo "🚀 Starting RevAI Pro AI Microservices..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.microservices.txt

# Create logs directory
mkdir -p logs

# Function to start a service
start_service() {
    local service_name=$1
    local port=$2
    local script=$3
    
    echo "🔄 Starting $service_name on port $port..."
    
    # Start service in background
    nohup python3 $script > logs/${service_name}.log 2>&1 &
    local pid=$!
    
    # Wait a moment for service to start
    sleep 2
    
    # Check if service is running
    if ps -p $pid > /dev/null; then
        echo "✅ $service_name started successfully (PID: $pid)"
        echo $pid > logs/${service_name}.pid
    else
        echo "❌ Failed to start $service_name"
        return 1
    fi
}

# Start all microservices
echo "🔧 Starting microservices..."

start_service "agent-registry" "8002" "src/microservices/agent_registry.py"
start_service "routing-orchestrator" "8003" "src/microservices/routing_orchestrator.py"
start_service "kpi-exporter" "8004" "src/microservices/kpi_exporter.py"
start_service "confidence-thresholds" "8005" "src/microservices/confidence_thresholds.py"
start_service "model-audit" "8006" "src/microservices/model_audit.py"
start_service "ai-orchestrator" "8001" "src/microservices/orchestrator.py"

echo ""
echo "🎉 All AI microservices started successfully!"
echo ""
echo "📊 Service Status:"
echo "  - AI Orchestrator:     http://localhost:8001"
echo "  - Agent Registry:      http://localhost:8002"
echo "  - Routing Orchestrator: http://localhost:8003"
echo "  - KPI Exporter:        http://localhost:8004"
echo "  - Confidence Thresholds: http://localhost:8005"
echo "  - Model Audit:         http://localhost:8006"
echo ""
echo "📝 Logs are available in the 'logs/' directory"
echo "🛑 To stop all services, run: ./stop-microservices.sh"
echo ""

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo "🔍 Checking service health..."
python3 -c "
import httpx
import asyncio

async def check_health():
    services = {
        'AI Orchestrator': 'http://localhost:8001',
        'Agent Registry': 'http://localhost:8002',
        'Routing Orchestrator': 'http://localhost:8003',
        'KPI Exporter': 'http://localhost:8004',
        'Confidence Thresholds': 'http://localhost:8005',
        'Model Audit': 'http://localhost:8006'
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(f'{url}/health')
                if response.status_code == 200:
                    print(f'✅ {name}: Healthy')
                else:
                    print(f'⚠️  {name}: Unhealthy (HTTP {response.status_code})')
            except Exception as e:
                print(f'❌ {name}: Unreachable ({str(e)})')

asyncio.run(check_health())
"

echo ""
echo "🎯 AI Microservices are ready for use!"
echo "   Visit http://localhost:8001/docs for API documentation"
