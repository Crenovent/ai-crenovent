#!/usr/bin/env python3
"""
Complete System Startup Script
=============================

This script provides instructions and utilities to run all three components:
1. AI Backend (FastAPI - Pipeline Agents)
2. Node.js Backend (Policy Management)
3. Frontend (Next.js - Pipeline Agents UI)
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

def print_banner():
    print("""
🚀 RevAI Pro - Complete System Startup Guide
============================================

This will help you start all three components:
1. 🤖 AI Backend (Pipeline Agents) - Port 8000
2. ⚙️  Node.js Backend (Policy Management) - Port 3001
3. 🎨 Frontend (Pipeline Agents UI) - Port 3000

""")

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2-binary', 
        'asyncpg', 'openai', 'pydantic', 'python-dotenv', 'requests', 'aiohttp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing Python packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All Python dependencies are installed")
    return True

def start_ai_backend():
    """Start the AI backend (FastAPI)"""
    print("\n🤖 Starting AI Backend (Pipeline Agents)...")
    print("   Port: 8000")
    print("   Command: python main.py")
    print("   Docs: http://localhost:8000/docs")
    
    # Check if we can start the server
    if not check_python_dependencies():
        return False
    
    try:
        # Start the server
        print("\n   Starting FastAPI server...")
        os.system("python main.py")
    except KeyboardInterrupt:
        print("\n   AI Backend stopped by user")
    except Exception as e:
        print(f"   ❌ Failed to start AI Backend: {e}")
        return False
    
    return True

def print_nodejs_instructions():
    """Print instructions for starting Node.js backend"""
    print("""
⚙️  Node.js Backend (Policy Management) - Manual Start Required
==============================================================

1. Open a new terminal
2. Navigate to: cd ../crenovent-backend
3. Install dependencies: npm install
4. Start server: npm start (or node server.js)
5. Server will run on: http://localhost:3001

This backend handles:
- Pipeline policy management
- User authentication
- Database connections
- Policy CRUD operations

""")

def print_frontend_instructions():
    """Print instructions for starting Frontend"""
    print("""
🎨 Frontend (Pipeline Agents UI) - Manual Start Required
========================================================

1. Open a new terminal
2. Navigate to: cd ../portal-crenovent
3. Install dependencies: npm install (or yarn install)
4. Start development server: npm run dev (or yarn dev)
5. Open browser: http://localhost:3000
6. Navigate to: /dashboard/user/pipeline-agents

Features available:
- Policy-aware pipeline agents
- Natural language queries
- Real-time policy compliance
- Agent composition visualization
- Execution history and testing

""")

def check_system_health():
    """Check if all systems are running"""
    print("\n🔍 System Health Check")
    print("=" * 50)
    
    # Check AI Backend
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ AI Backend (Port 8000): HEALTHY")
            data = response.json()
            print(f"   - Atomic Agents: {data.get('system_metrics', {}).get('atomic_agents_count', 0)}")
            print(f"   - Policy Integration: {data.get('system_metrics', {}).get('policy_integration', 'unknown')}")
        else:
            print("⚠️  AI Backend (Port 8000): DEGRADED")
    except:
        print("❌ AI Backend (Port 8000): NOT RUNNING")
    
    # Check Node.js Backend
    try:
        response = requests.get("http://localhost:3001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Node.js Backend (Port 3001): HEALTHY")
        else:
            print("⚠️  Node.js Backend (Port 3001): DEGRADED")
    except:
        print("❌ Node.js Backend (Port 3001): NOT RUNNING")
    
    # Check Frontend
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend (Port 3000): HEALTHY")
        else:
            print("⚠️  Frontend (Port 3000): DEGRADED")
    except:
        print("❌ Frontend (Port 3000): NOT RUNNING")

def main():
    print_banner()
    
    choice = input("Choose an option:\n1. Start AI Backend only\n2. Show all startup instructions\n3. Check system health\n\nEnter choice (1-3): ")
    
    if choice == "1":
        start_ai_backend()
    elif choice == "2":
        print_nodejs_instructions()
        print_frontend_instructions()
        print("\n💡 Recommended startup order:")
        print("1. Start Node.js Backend first (for policy management)")
        print("2. Start AI Backend second (for pipeline agents)")
        print("3. Start Frontend last (for UI)")
        print("\nOnce all are running, test at: http://localhost:3000/dashboard/user/pipeline-agents")
    elif choice == "3":
        check_system_health()
    else:
        print("Invalid choice. Run again with a valid option.")

if __name__ == "__main__":
    main()
