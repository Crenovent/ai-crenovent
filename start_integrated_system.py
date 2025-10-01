#!/usr/bin/env python3
"""
Start Integrated System
Launches the complete RBA system with all components properly connected
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add current directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def start_system():
    """Start the complete integrated system"""
    
    print("🚀 STARTING REVAI PRO INTEGRATED SYSTEM")
    print("=" * 50)
    print(f"⏰ Startup Time: {datetime.now().isoformat()}")
    print()
    
    try:
        # 1. Load environment variables
        print("1️⃣ Loading Environment Configuration...")
        from dotenv import load_dotenv
        load_dotenv()
        
        # Verify critical environment variables
        required_vars = [
            'DATABASE_URL',
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file")
            return False
        
        print("✅ Environment configuration loaded")
        
        # 2. Initialize Integration Orchestrator
        print("\n2️⃣ Initializing Integration Orchestrator...")
        from dsl.integration_orchestrator import get_integration_orchestrator
        
        orchestrator = await get_integration_orchestrator()
        status = await orchestrator.get_system_status()
        
        if status['status'] == 'operational':
            print("✅ Integration Orchestrator operational")
            print(f"🔧 Components: {len(status['components'])}")
            
            # Display component status
            print("\n📊 Component Status:")
            for component in status['components']:
                print(f"   ✅ {component}")
                
            # Display capabilities
            print("\n🌟 System Capabilities:")
            capabilities = status.get('capabilities', {})
            for capability, enabled in capabilities.items():
                icon = "✅" if enabled else "❌"
                print(f"   {icon} {capability.replace('_', ' ').title()}")
                
        else:
            print(f"❌ Integration Orchestrator failed to start: {status}")
            return False
        
        # 3. Test system functionality
        print("\n3️⃣ Testing System Functionality...")
        
        test_result = await orchestrator.route_and_execute(
            user_input="test system startup",
            tenant_id=1300,
            user_id=1319,
            context_data={"startup_test": True}
        )
        
        if test_result['status'] == 'success':
            print("✅ System functionality test passed")
        else:
            print(f"⚠️ System test warning: {test_result.get('error', 'Unknown issue')}")
        
        # 4. Display system information
        print("\n4️⃣ System Information:")
        print("=" * 30)
        print(f"🏢 Multi-Tenant Support: {'✅ Enabled' if capabilities.get('multi_tenant_isolation') else '❌ Disabled'}")
        print(f"🧠 LLM/NLP Integration: {'✅ Active' if capabilities.get('llm_nlp_enabled') else '❌ Inactive'}")
        print(f"💬 Conversational Mode: {'✅ Available' if capabilities.get('conversational_mode') else '❌ Unavailable'}")
        print(f"📊 Knowledge Graph: {'✅ Active' if capabilities.get('knowledge_graph_enabled') else '❌ Inactive'}")
        print(f"🎯 Trust Scoring: {'✅ Monitoring' if capabilities.get('trust_scoring_enabled') else '❌ Disabled'}")
        print(f"📈 SLA Monitoring: {'✅ Active' if capabilities.get('sla_monitoring_enabled') else '❌ Inactive'}")
        print(f"🔄 Dynamic Capabilities: {'✅ Adapting' if capabilities.get('dynamic_capabilities') else '❌ Static'}")
        
        # 5. API Information
        print("\n5️⃣ API Endpoints Available:")
        print("=" * 30)
        print("🌐 Main API: http://localhost:8000")
        print("📋 Health Check: http://localhost:8000/health")
        print("🧠 Intent Parsing: POST http://localhost:8000/api/workflow-builder/parse-intent")
        print("🚀 Natural Language Execution: POST http://localhost:8000/api/workflow-builder/execute-natural-language")
        print("🔍 Workflow Discovery: GET http://localhost:8000/api/hub/workflows/discover")
        print("📊 Hub Analytics: GET http://localhost:8000/api/hub/analytics/dashboard")
        print("🏛️ Capability Registry: GET http://localhost:8000/api/capability-registry/")
        
        # 6. Usage Examples
        print("\n6️⃣ Usage Examples:")
        print("=" * 20)
        print("💬 Try these natural language queries:")
        print('   • "Show me pipeline opportunities"')
        print('   • "Calculate sales commissions"') 
        print('   • "Check forecast accuracy"')
        print('   • "Find stale deals"')
        print('   • "Generate account health report"')
        
        print("\n🎉 SYSTEM STARTUP COMPLETE!")
        print("🚀 RevAI Pro is ready for automation requests")
        print("📖 Use the API endpoints above or run test_complete_system_integration.py for testing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_interactive_mode():
    """Run interactive mode for testing"""
    
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 20)
    print("Enter natural language queries to test the system.")
    print("Type 'quit' to exit.")
    print()
    
    from dsl.integration_orchestrator import get_integration_orchestrator
    orchestrator = await get_integration_orchestrator()
    
    while True:
        try:
            user_input = input("🗣️  You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤖 Processing...")
            
            result = await orchestrator.route_and_execute(
                user_input=user_input,
                tenant_id=1300,
                user_id=1319,
                context_data={"interactive": True}
            )
            
            if result['status'] == 'success':
                routing = result.get('routing_result', {})
                execution = result.get('execution_result', {})
                
                print(f"✅ Intent: {routing.get('intent_type', 'unknown')}")
                print(f"🎯 Capability: {routing.get('selected_capability', 'unknown')}")
                print(f"🤖 Type: {routing.get('automation_type', 'unknown')}")
                print(f"📊 Confidence: {routing.get('confidence', 0):.2f}")
                print(f"⚡ Status: {execution.get('status', 'unknown')}")
                
                if result.get('llm_reasoning'):
                    print(f"💭 Reasoning: {result['llm_reasoning'][:200]}...")
                    
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Start system first
        success = await start_system()
        if success:
            await run_interactive_mode()
    else:
        # Just start the system
        await start_system()

if __name__ == "__main__":
    asyncio.run(main())
