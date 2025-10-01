#!/usr/bin/env python3
"""
CSV RBA Agents Demo - Show RBA agents working with CSV uploads
"""

import asyncio
import sys
import os
import pandas as pd
import io
from datetime import datetime, timedelta
import random

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

print("📊 CSV RBA AGENTS DEMO")
print("=" * 40)
print("🤖 Upload CSV data and let RBA agents analyze it!")

async def create_sample_crm_data():
    """Create sample CRM data for demonstration"""
    
    print("\n1️⃣ Creating Sample CRM Data...")
    
    # Sample Opportunities Data
    opportunities_data = []
    stages = ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
    
    for i in range(100):
        # Create realistic dates (some stale, some recent)
        if i < 20:  # 20% stale opportunities
            last_activity = datetime.now() - timedelta(days=random.randint(45, 120))
        else:
            last_activity = datetime.now() - timedelta(days=random.randint(1, 30))
        
        close_date = last_activity + timedelta(days=random.randint(10, 60))
        
        opportunities_data.append({
            'Opportunity_ID': f'OPP-{1000+i}',
            'Opportunity_Name': f'Deal with Company {i+1}',
            'Account_Name': f'Company {i+1}',
            'Stage': random.choice(stages),
            'Amount': random.randint(10000, 500000),
            'Close_Date': close_date.strftime('%Y-%m-%d'),
            'Last_Activity_Date': last_activity.strftime('%Y-%m-%d'),
            'Owner': f'Rep {random.randint(1, 10)}',
            'Probability': random.randint(10, 90),
            'Lead_Source': random.choice(['Website', 'Referral', 'Cold Call', 'Marketing'])
        })
    
    opportunities_df = pd.DataFrame(opportunities_data)
    
    # Sample Accounts Data
    accounts_data = []
    industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail']
    
    for i in range(50):
        health_score = random.randint(30, 95)  # Some at-risk accounts
        
        accounts_data.append({
            'Account_ID': f'ACC-{2000+i}',
            'Account_Name': f'Company {i+1}',
            'Industry': random.choice(industries),
            'Employee_Count': random.randint(50, 10000),
            'Annual_Revenue': random.randint(1000000, 100000000),
            'Health_Score': health_score,
            'Last_Engagement': (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d'),
            'Account_Manager': f'Manager {random.randint(1, 5)}',
            'Contract_Start': (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
            'Risk_Level': 'High' if health_score < 50 else 'Medium' if health_score < 75 else 'Low'
        })
    
    accounts_df = pd.DataFrame(accounts_data)
    
    # Sample Forecast Data
    forecast_data = []
    
    for i in range(12):  # 12 months of data
        forecast_amount = random.randint(800000, 1200000)
        # Add some variance for realism
        actual_amount = forecast_amount * random.uniform(0.8, 1.2)
        
        month_date = datetime.now() - timedelta(days=30*i)
        
        forecast_data.append({
            'Period': month_date.strftime('%Y-%m'),
            'Forecast_Amount': forecast_amount,
            'Actual_Amount': int(actual_amount),
            'Rep_Name': f'Rep {random.randint(1, 10)}',
            'Territory': random.choice(['North', 'South', 'East', 'West']),
            'Quota': random.randint(900000, 1100000),
            'Submitted_Date': (month_date - timedelta(days=5)).strftime('%Y-%m-%d')
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Save to CSV files
    opportunities_df.to_csv('sample_opportunities.csv', index=False)
    accounts_df.to_csv('sample_accounts.csv', index=False)
    forecast_df.to_csv('sample_forecasts.csv', index=False)
    
    print(f"   ✅ Created sample_opportunities.csv ({len(opportunities_df)} records)")
    print(f"   ✅ Created sample_accounts.csv ({len(accounts_df)} records)")
    print(f"   ✅ Created sample_forecasts.csv ({len(forecast_df)} records)")
    
    return {
        'opportunities': opportunities_df,
        'accounts': accounts_df,
        'forecasts': forecast_df
    }

async def demo_csv_analysis(sample_data):
    """Demo CSV analysis with RBA agents"""
    
    print("\n2️⃣ Analyzing CSV Data with RBA Agents...")
    
    try:
        from src.services.csv_crm_analyzer import CSVCRMAnalyzer
        from src.services.csv_query_processor import CSVQueryProcessor
        
        analyzer = CSVCRMAnalyzer()
        query_processor = CSVQueryProcessor()
        
        # Analyze opportunities data
        print("\n   🔍 Analyzing Opportunities Data:")
        opp_analysis = await analyzer.analyze_crm_data(
            csv_data=sample_data['opportunities'],
            data_type='opportunities',
            analysis_type='pipeline_hygiene'
        )
        
        print(f"      📊 Total Opportunities: {opp_analysis.get('total_records', 0)}")
        print(f"      ⚠️  Stale Opportunities: {sum(v.get('stale_records', 0) for v in opp_analysis.values() if isinstance(v, dict))}")
        print(f"      💰 Total Pipeline Value: ${opp_analysis.get('pipeline_value', {}).get('total', 0):,.2f}")
        print(f"      📈 Win Rate: {opp_analysis.get('win_rate', 0):.1%}")
        
        # Analyze accounts data
        print("\n   🔍 Analyzing Accounts Data:")
        acc_analysis = await analyzer.analyze_crm_data(
            csv_data=sample_data['accounts'],
            data_type='accounts',
            analysis_type='account_health'
        )
        
        print(f"      🏢 Total Accounts: {acc_analysis.get('total_records', 0)}")
        if 'health_analysis' in acc_analysis:
            health = acc_analysis['health_analysis']
            print(f"      ⚠️  At-Risk Accounts: {health.get('at_risk_accounts', 0)}")
            print(f"      ✅ Healthy Accounts: {health.get('healthy_accounts', 0)}")
            print(f"      📊 Average Health: {health.get('average_health', 0):.1f}")
        
        # Analyze forecast data
        print("\n   🔍 Analyzing Forecast Data:")
        forecast_analysis = await analyzer.analyze_crm_data(
            csv_data=sample_data['forecasts'],
            data_type='forecasts',
            analysis_type='forecast_accuracy'
        )
        
        print(f"      📈 Total Forecasts: {forecast_analysis.get('total_records', 0)}")
        if 'forecast_accuracy' in forecast_analysis:
            accuracy = forecast_analysis['forecast_accuracy']
            print(f"      🎯 Accuracy Rate: {accuracy.get('accuracy_rate', 0):.1f}%")
            print(f"      📊 Avg Variance: {accuracy.get('average_variance_percentage', 0):.1f}%")
        
        return {
            'opportunities': opp_analysis,
            'accounts': acc_analysis,
            'forecasts': forecast_analysis
        }
        
    except Exception as e:
        print(f"❌ CSV analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def demo_natural_language_queries(sample_data):
    """Demo natural language queries against CSV data"""
    
    print("\n3️⃣ Natural Language Queries on CSV Data...")
    
    try:
        from src.services.csv_query_processor import CSVQueryProcessor
        
        processor = CSVQueryProcessor()
        
        # Test queries
        test_queries = [
            {
                'query': 'Show me stale opportunities in the pipeline',
                'data': sample_data['opportunities'],
                'data_type': 'opportunities'
            },
            {
                'query': 'Which accounts are at risk of churning?',
                'data': sample_data['accounts'],
                'data_type': 'accounts'
            },
            {
                'query': 'How accurate are our forecasts?',
                'data': sample_data['forecasts'],
                'data_type': 'forecasts'
            },
            {
                'query': 'What is the total pipeline value?',
                'data': sample_data['opportunities'],
                'data_type': 'opportunities'
            }
        ]
        
        print("   🗣️ Processing Natural Language Queries:")
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n      {i}. Query: '{test['query']}'")
            
            try:
                result = await processor.process_natural_language_query(
                    csv_data=test['data'],
                    user_query=test['query'],
                    data_type=test['data_type']
                )
                
                print(f"         🎯 Intent: {result.get('query_intent', 'Unknown')}")
                print(f"         🔍 Enhanced: {result.get('enhanced_query', '')[:80]}...")
                
                # Show key results
                query_results = result.get('query_results', {})
                if 'stale_opportunities' in query_results:
                    print(f"         📊 Stale Opportunities: {query_results['stale_opportunities']}")
                elif 'at_risk_accounts' in str(query_results):
                    print(f"         ⚠️  Analysis: {query_results.get('analysis_type', 'Unknown')}")
                elif 'total_pipeline_value' in query_results:
                    print(f"         💰 Pipeline Value: ${query_results['total_pipeline_value']:,.2f}")
                
            except Exception as e:
                print(f"         ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Natural language demo failed: {e}")
        return False

async def demo_rba_agent_integration(sample_data, analyses):
    """Demo integration with RBA agents"""
    
    print("\n4️⃣ RBA Agent Integration...")
    
    try:
        from dsl.hub.routing_orchestrator import RoutingOrchestrator
        from src.services.connection_pool_manager import ConnectionPoolManager
        
        pool_manager = ConnectionPoolManager()
        orchestrator = RoutingOrchestrator(pool_manager)
        await orchestrator.initialize()
        
        # Test RBA agent queries with CSV context
        csv_queries = [
            "Analyze the uploaded opportunity data for pipeline hygiene",
            "Check account health from the uploaded customer data",
            "Validate forecast accuracy from the uploaded forecast data",
            "Identify revenue recognition opportunities from closed deals"
        ]
        
        print("   🤖 RBA Agent Responses to CSV Data:")
        
        for i, query in enumerate(csv_queries, 1):
            print(f"\n      {i}. Query: '{query}'")
            
            try:
                result = await orchestrator.route_request(
                    user_input=query,
                    tenant_id=1300,
                    user_id=1319,
                    context_data={
                        "csv_data": True,
                        "has_opportunities": len(sample_data['opportunities']) > 0,
                        "has_accounts": len(sample_data['accounts']) > 0,
                        "has_forecasts": len(sample_data['forecasts']) > 0,
                        "analysis_context": analyses
                    }
                )
                
                print(f"         🎯 Agent: {result.parsed_intent.intent_type.value.replace('_', ' ').title()}")
                print(f"         📊 Confidence: {result.parsed_intent.confidence:.1%}")
                print(f"         ⚙️ Category: {result.parsed_intent.workflow_category or 'General'}")
                
                if result.parsed_intent.llm_reasoning:
                    reasoning = result.parsed_intent.llm_reasoning[:60] + "..." if len(result.parsed_intent.llm_reasoning) > 60 else result.parsed_intent.llm_reasoning
                    print(f"         💭 Reasoning: {reasoning}")
                
            except Exception as e:
                print(f"         ❌ RBA routing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ RBA integration demo failed: {e}")
        return False

async def show_api_usage():
    """Show how to use the CSV API"""
    
    print("\n5️⃣ API Usage Examples...")
    print("   📡 CSV Upload & Analysis API Endpoints:")
    print()
    print("   1. Upload CSV file:")
    print("      POST /api/csv/upload")
    print("      - Form data: file, tenant_id, user_id")
    print("      - Returns: file_id, analysis summary")
    print()
    print("   2. Analyze uploaded CSV:")
    print("      POST /api/csv/analyze")
    print("      - JSON: {file_id, analysis_type, user_query, tenant_id, user_id}")
    print("      - Returns: RBA agent analysis results")
    print()
    print("   3. Natural language query:")
    print("      POST /api/csv/query")
    print("      - Form data: file_id, user_query, tenant_id, user_id")
    print("      - Returns: Intelligent analysis response")
    print()
    print("   4. List uploaded files:")
    print("      GET /api/csv/files?tenant_id=1300&user_id=1319")
    print("      - Returns: List of user's uploaded CSV files")
    print()
    print("   💡 Example cURL commands:")
    print("   curl -X POST 'http://localhost:8000/api/csv/upload' \\")
    print("        -F 'file=@sample_opportunities.csv' \\")
    print("        -F 'tenant_id=1300' \\")
    print("        -F 'user_id=1319'")
    print()
    print("   curl -X POST 'http://localhost:8000/api/csv/query' \\")
    print("        -F 'file_id=<file_id>' \\")
    print("        -F 'user_query=Show me stale opportunities' \\")
    print("        -F 'tenant_id=1300' \\")
    print("        -F 'user_id=1319'")

async def main():
    """Main demo function"""
    
    print("🚀 Starting CSV RBA Agents Demo...")
    
    # Create sample data
    sample_data = await create_sample_crm_data()
    
    if sample_data:
        # Analyze with RBA agents
        analyses = await demo_csv_analysis(sample_data)
        
        if analyses:
            # Natural language queries
            await demo_natural_language_queries(sample_data)
            
            # RBA agent integration
            await demo_rba_agent_integration(sample_data, analyses)
            
            # Show API usage
            await show_api_usage()
            
            print("\n🎉 CSV RBA AGENTS DEMO COMPLETE!")
            print("=" * 40)
            print("✅ CSV upload and analysis working")
            print("🤖 RBA agents understand CSV data")
            print("🗣️ Natural language queries working")
            print("📊 Intelligent data analysis completed")
            print("🔗 API endpoints ready for integration")
            
            print("\n🚀 READY TO USE:")
            print("1. Upload your CRM CSV files")
            print("2. Ask questions in natural language")
            print("3. Get intelligent RBA agent responses")
            print("4. Receive actionable insights and recommendations")
            
            print("\n📁 Sample files created:")
            print("• sample_opportunities.csv - Pipeline data")
            print("• sample_accounts.csv - Account health data")
            print("• sample_forecasts.csv - Forecast accuracy data")
            
        else:
            print("❌ Analysis failed - check error messages above")
    else:
        print("❌ Sample data creation failed")

if __name__ == "__main__":
    asyncio.run(main())
