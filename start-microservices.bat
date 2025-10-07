@echo off
REM AI Microservices Startup Script for Windows
REM Starts all AI microservices with proper configuration

echo 🚀 Starting RevAI Pro AI Microservices...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is required but not installed.
    exit /b 1
)

REM Install requirements
echo 📦 Installing Python dependencies...
pip install -r requirements.microservices.txt

REM Create logs directory
if not exist logs mkdir logs

REM Function to start a service
:start_service
set service_name=%1
set port=%2
set script=%3

echo 🔄 Starting %service_name% on port %port%...

REM Start service in background
start /b python %script% > logs\%service_name%.log 2>&1

REM Wait a moment for service to start
timeout /t 2 /nobreak >nul

echo ✅ %service_name% started successfully
goto :eof

REM Start all microservices
echo 🔧 Starting microservices...

call :start_service "agent-registry" "8002" "src/microservices/agent_registry.py"
call :start_service "routing-orchestrator" "8003" "src/microservices/routing_orchestrator.py"
call :start_service "kpi-exporter" "8004" "src/microservices/kpi_exporter.py"
call :start_service "confidence-thresholds" "8005" "src/microservices/confidence_thresholds.py"
call :start_service "model-audit" "8006" "src/microservices/model_audit.py"
call :start_service "calendar-automation" "8007" "src/microservices/calendar_automation.py"
call :start_service "letsmeet-automation" "8008" "src/microservices/letsmeet_automation.py"
call :start_service "cruxx-automation" "8009" "src/microservices/cruxx_automation.py"
call :start_service "run-trace-schema" "8010" "src/microservices/run_trace_schema.py"
call :start_service "dlq-replay-tooling" "8011" "src/microservices/dlq_replay_tooling.py"
call :start_service "metrics-exporter" "8012" "src/microservices/metrics_exporter.py"
call :start_service "event-bus-schema-registry" "8013" "src/microservices/event_bus_schema_registry.py"
call :start_service "ai-orchestrator" "8001" "src/microservices/orchestrator.py"

echo.
echo 🎉 All AI microservices started successfully!
echo.
echo 📊 Service Status:
echo   - AI Orchestrator:     http://localhost:8001
echo   - Agent Registry:      http://localhost:8002
echo   - Routing Orchestrator: http://localhost:8003
echo   - KPI Exporter:        http://localhost:8004
echo   - Confidence Thresholds: http://localhost:8005
echo   - Model Audit:         http://localhost:8006
echo   - Calendar Automation: http://localhost:8007
echo   - Let's Meet Automation: http://localhost:8008
echo   - Cruxx Automation:   http://localhost:8009
echo   - Run Trace Schema:    http://localhost:8010
echo   - DLQ + Replay Tooling: http://localhost:8011
echo   - Metrics Exporter:    http://localhost:8012
echo   - Event Bus + Schema:  http://localhost:8013
echo.
echo 📝 Logs are available in the 'logs/' directory
echo 🛑 To stop all services, press Ctrl+C
echo.

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 5 /nobreak >nul

echo 🎯 AI Microservices are ready for use!
echo    Visit http://localhost:8001/docs for API documentation

pause
