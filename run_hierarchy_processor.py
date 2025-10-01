#!/usr/bin/env python3
"""
Hierarchy Processor Runner
Simple script to start the hierarchy processor service
"""

import uvicorn
import logging
import sys
import os

# Add the current directory to Python path so we can import hierarchy_processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the service
    print("🚀 Starting Hierarchy Processor Service...")
    print("📡 API will be available at: http://localhost:8001")
    print("📚 API docs available at: http://localhost:8001/docs")
    print("🏥 Health check: http://localhost:8001/health")
    print("🔄 Normalizes HRMS CSV exports for Node.js backend integration")
    
    uvicorn.run(
        "hierarchy_processor.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
