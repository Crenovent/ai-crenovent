"""
Intelligent Query Engine
Handles complex questions for strategic planning data using AI reasoning with pgvector
"""

import logging
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncAzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class IntelligentQueryEngine:
    """
    Advanced query engine for strategic planning data that uses AI to:
    1. Classify planning-related queries
    2. Analyze strategic relationships and patterns
    3. Generate planning insights from pgvector data
    4. Handle complex strategic reasoning
    """
    
    def __init__(self):
        self.openai_client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version
        )
        
        # Query classification patterns
        self.query_patterns = {
            'analytical': [
                r'analyze|analysis|trend|pattern|insight|correlation',
                r'compare|comparison|versus|vs|difference',
                r'performance|efficiency|optimization',
                r'prediction|forecast|future|expect'
            ],
            'statistical': [
                r'average|mean|median|mode|standard deviation',
                r'percentage|ratio|proportion|rate',
                r'distribution|variance|correlation',
                r'statistical|stats|metrics'
            ],
            'strategic': [
                r'recommend|suggestion|advice|should|strategy',
                r'best|optimal|improve|optimize|enhance',
                r'risk|opportunity|potential|growth',
                r'decision|choice|option|alternative'
            ],
            'explanatory': [
                r'why|how|what.*mean|explain|because',
                r'reason|cause|factor|influence|impact',
                r'understand|clarify|elaborate|detail'
            ]
        }
    
    async def process_intelligent_query(self, question: str, csv_data_summary: Dict, 
                                      search_results: List[Dict]) -> Dict[str, Any]:
        """
        Process complex queries using AI intelligence
        """
        try:
            # Step 1: Classify the query type
            query_type = await self._classify_query(question)
            
            # Step 2: Generate intelligent response based on type
            if query_type == 'analytical':
                response = await self._handle_analytical_query(question, csv_data_summary, search_results)
            elif query_type == 'statistical':
                response = await self._handle_statistical_query(question, csv_data_summary, search_results)
            elif query_type == 'strategic':
                response = await self._handle_strategic_query(question, csv_data_summary, search_results)
            elif query_type == 'explanatory':
                response = await self._handle_explanatory_query(question, csv_data_summary, search_results)
            else:
                # Default intelligent processing
                response = await self._handle_general_intelligent_query(question, csv_data_summary, search_results)
            
            return {
                "answer": response,
                "query_type": query_type,
                "intelligence_level": "advanced",
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"❌ Intelligent query processing failed: {e}")
            return {
                "answer": "I encountered an error while processing your complex query. Please try rephrasing it.",
                "query_type": "error",
                "intelligence_level": "basic",
                "confidence": 0.1
            }
    
    async def _classify_query(self, question: str) -> str:
        """Classify the type of query using pattern matching and AI"""
        
        question_lower = question.lower()
        
        # Pattern-based classification
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return query_type
        
        # AI-based classification for complex cases
        try:
            prompt = f"""
            Classify this business query into one of these categories:
            - analytical: Seeking patterns, trends, insights, comparisons
            - statistical: Asking for calculations, averages, distributions
            - strategic: Requesting recommendations, advice, optimization
            - explanatory: Asking why/how something works or what it means
            - direct: Simple data retrieval question
            
            Query: "{question}"
            
            Return only the category name:
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            if classification in self.query_patterns.keys() or classification == 'direct':
                return classification
            else:
                return 'general'
                
        except Exception:
            return 'general'
    
    async def _handle_analytical_query(self, question: str, data_summary: Dict, 
                                     search_results: List[Dict]) -> str:
        """Handle analytical queries with trend analysis and insights"""
        
        prompt = f"""
        You are a senior business analyst. Analyze the following data and provide strategic insights.
        
        Data Context:
        {self._safe_json_dumps(data_summary)}
        
        Search Results:
        {self._format_search_results(search_results)}
        
        Question: {question}
        
        Provide a comprehensive analytical response that includes:
        1. Key findings and patterns
        2. Business implications
        3. Trends or correlations identified
        4. Actionable insights
        
        Be specific and use data to support your analysis:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=100)
    
    async def _handle_statistical_query(self, question: str, data_summary: Dict, 
                                      search_results: List[Dict]) -> str:
        """Handle statistical analysis queries"""
        
        prompt = f"""
        You are a data scientist. Provide statistical analysis for this business question.
        
        Data Available:
        {self._safe_json_dumps(data_summary)}
        
        Search Results:
        {self._format_search_results(search_results)}
        
        Question: {question}
        
        Provide statistical analysis including:
        1. Relevant calculations and metrics
        2. Statistical significance if applicable
        3. Data distribution insights
        4. Confidence intervals or error margins where appropriate
        
        Use precise numbers and statistical terminology:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=150)
    
    async def _handle_strategic_query(self, question: str, data_summary: Dict, 
                                    search_results: List[Dict]) -> str:
        """Handle strategic and recommendation queries"""
        
        prompt = f"""
        You are a business strategy consultant. Provide strategic recommendations based on the data.
        
        Business Data:
        {self._safe_json_dumps(data_summary)}
        
        Relevant Information:
        {self._format_search_results(search_results)}
        
        Question: {question}
        
        Provide strategic guidance including:
        1. Specific recommendations
        2. Rationale based on data
        3. Potential risks and opportunities
        4. Implementation priorities
        5. Expected outcomes
        
        Be actionable and business-focused:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=100)
    
    async def _handle_explanatory_query(self, question: str, data_summary: Dict, 
                                      search_results: List[Dict]) -> str:
        """Handle why/how explanatory queries"""
        
        prompt = f"""
        You are a business intelligence expert. Explain the underlying reasons and mechanisms.
        
        Data Context:
        {self._safe_json_dumps(data_summary)}
        
        Supporting Information:
        {self._format_search_results(search_results)}
        
        Question: {question}
        
        Provide a clear explanation that includes:
        1. Root causes or mechanisms
        2. Contributing factors
        3. Business context and implications
        4. Examples from the data if relevant
        
        Make it easy to understand while being thorough:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=150)
    
    async def _handle_general_intelligent_query(self, question: str, data_summary: Dict, 
                                              search_results: List[Dict]) -> str:
        """Handle general complex queries with intelligence"""
        
        prompt = f"""
        You are a concise business analyst. Answer ONLY what the user asks for.
        
        Available Data:
        {self._safe_json_dumps(data_summary)}
        
        Relevant Search Results:
        {self._format_search_results(search_results)}
        
        Question: {question}
        
        CRITICAL RULES:
        - Give ONLY the direct answer
        - For counts: Just state the number (e.g., "1,050 opportunities")
        - For lists: Use bullet points only
        - NO explanations, NO analysis, NO extra context
        - Be extremely concise
        
        Answer:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=100)
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for AI context"""
        
        if not search_results:
            return "No specific data found for this query."
        
        formatted = []
        for i, result in enumerate(search_results[:5], 1):  # Top 5 results
            content = result.get('content', 'No content')
            similarity = result.get('similarity', 0)
            formatted.append(f"Result {i} (Relevance: {similarity:.2f}):\n{content}")
        
        return '\n\n'.join(formatted)
    
    async def _generate_ai_response(self, prompt: str, max_tokens: int = 400) -> str:
        """Generate AI response with error handling"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=[
                    {"role": "system", "content": "You are a concise business analyst. Answer ONLY what is asked. Use bullet points for lists. Be brief and direct."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ AI response generation failed: {e}")
            return "I apologize, but I encountered an error while generating an intelligent response. Please try rephrasing your question."
    
    def should_use_intelligence(self, question: str) -> bool:
        """Determine if a query needs intelligent processing"""
        
        # Keywords that indicate complex queries (NOT simple counts)
        intelligent_keywords = [
            'why', 'analyze', 'recommend', 'suggest', 'compare', 
            'trend', 'pattern', 'insight', 'strategy', 'optimal', 'best',
            'improve', 'optimize', 'predict', 'forecast', 'explain',
            'correlation', 'relationship', 'impact', 'influence', 'cause'
        ]
        
        # Simple count queries should NOT use intelligence
        simple_count_patterns = [
            'how many', 'count', 'total', 'number of'
        ]
        
        question_lower = question.lower()
        
        # Skip intelligence for simple counts
        for pattern in simple_count_patterns:
            if pattern in question_lower and len(question.split()) <= 6:
                return False
        
        # Check for intelligent keywords
        for keyword in intelligent_keywords:
            if keyword in question_lower:
                return True
        
        # Check for list requests that need formatting
        list_keywords = ['list', 'show me', 'give me', 'display', 'breakdown']
        for keyword in list_keywords:
            if keyword in question_lower:
                return True
        
        # Check for complex sentence structures
        if len(question.split()) > 8:  # Long questions often need intelligence
            return True
        
        # Check for question words that indicate complexity
        complex_starters = ['what if', 'how can', 'why do', 'what should', 'which is better']
        for starter in complex_starters:
            if question_lower.startswith(starter):
                return True
        
        return False
    
    async def get_data_insights(self, data_summary: Dict) -> str:
        """Generate automatic insights about the data"""
        
        prompt = f"""
        Analyze this business data and provide 3-5 key insights that would be valuable to business stakeholders:
        
        Data Summary:
        {self._safe_json_dumps(data_summary)}
        
        Provide insights in this format:
        • Insight 1: [specific finding with numbers]
        • Insight 2: [business implication]
        • Insight 3: [opportunity or recommendation]
        
        Focus on actionable business intelligence:
        """
        
        return await self._generate_ai_response(prompt, max_tokens=100)
    
    def _safe_json_dumps(self, data: Any) -> str:
        """Safely serialize data to JSON, handling datetime objects"""
        
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        try:
            return json.dumps(data, indent=2, default=default_serializer)
        except Exception as e:
            logger.warning(f"JSON serialization failed: {e}")
            return str(data)
