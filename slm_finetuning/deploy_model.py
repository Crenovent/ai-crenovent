#!/usr/bin/env python3
"""
Deploy fine-tuned SLM for production use in account planning system
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, List
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanningExtractorSLM:
    def __init__(self, model_path: str = "phi3-planning-extractor"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load the fine-tuned model for inference"""
        logger.info(f"🚀 Loading fine-tuned model from: {self.model_path}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                return_full_text=False
            )
            
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def extract_planning_data(self, user_message: str) -> Dict[str, Any]:
        """Extract structured planning data from user message"""
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format prompt for the fine-tuned model
        prompt = f"""<|system|>
You are an expert AI assistant specialized in extracting structured information from business account planning requests. Extract all relevant information and return it as a properly formatted JSON object.

<|user|>
{user_message}

<|assistant|>
"""
        
        start_time = time.time()
        
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]['generated_text'].strip()
            
            # Try to parse as JSON
            try:
                # Clean up response - remove any markdown formatting
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                extracted_data = json.loads(response)
                
                # Add metadata
                extracted_data['_extraction_metadata'] = {
                    'model': 'phi3-planning-extractor',
                    'extraction_time_ms': round((time.time() - start_time) * 1000, 2),
                    'confidence': 'high',
                    'method': 'fine-tuned-slm'
                }
                
                logger.info(f"✅ Extraction completed in {extracted_data['_extraction_metadata']['extraction_time_ms']}ms")
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response}")
                
                # Fallback: return raw response with error info
                return {
                    'error': 'json_parse_failed',
                    'raw_response': response,
                    '_extraction_metadata': {
                        'model': 'phi3-planning-extractor',
                        'extraction_time_ms': round((time.time() - start_time) * 1000, 2),
                        'confidence': 'low',
                        'method': 'fine-tuned-slm'
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return {
                'error': 'extraction_failed',
                'error_details': str(e),
                '_extraction_metadata': {
                    'model': 'phi3-planning-extractor',
                    'extraction_time_ms': round((time.time() - start_time) * 1000, 2),
                    'confidence': 'failed',
                    'method': 'fine-tuned-slm'
                }
            }

    def batch_extract(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Extract planning data from multiple messages"""
        results = []
        
        logger.info(f"🔄 Processing {len(messages)} messages...")
        
        for i, message in enumerate(messages):
            logger.info(f"Processing message {i+1}/{len(messages)}")
            result = self.extract_planning_data(message)
            results.append(result)
        
        logger.info("✅ Batch processing completed!")
        return results

    def benchmark_performance(self, test_messages: List[str]) -> Dict[str, Any]:
        """Benchmark model performance"""
        logger.info("🏃 Running performance benchmark...")
        
        start_time = time.time()
        results = []
        successful_extractions = 0
        total_extraction_time = 0
        
        for message in test_messages:
            result = self.extract_planning_data(message)
            results.append(result)
            
            if 'error' not in result:
                successful_extractions += 1
                total_extraction_time += result.get('_extraction_metadata', {}).get('extraction_time_ms', 0)
        
        total_time = time.time() - start_time
        
        benchmark_results = {
            'total_messages': len(test_messages),
            'successful_extractions': successful_extractions,
            'success_rate': round((successful_extractions / len(test_messages)) * 100, 2),
            'average_extraction_time_ms': round(total_extraction_time / max(successful_extractions, 1), 2),
            'total_benchmark_time_s': round(total_time, 2),
            'throughput_messages_per_second': round(len(test_messages) / total_time, 2)
        }
        
        logger.info(f"📊 Benchmark Results:")
        logger.info(f"   Success Rate: {benchmark_results['success_rate']}%")
        logger.info(f"   Avg Extraction Time: {benchmark_results['average_extraction_time_ms']}ms")
        logger.info(f"   Throughput: {benchmark_results['throughput_messages_per_second']} msg/s")
        
        return benchmark_results

def main():
    # Test the deployed model
    extractor = PlanningExtractorSLM()
    
    # Load model
    extractor.load_model()
    
    # Test messages
    test_messages = [
        "Create account plan for GlobalTech Corp, customer since Q1 2024, revenue around 50M, highlighting renewal risks and expansion opportunities, schedule QBR on March 15 2025, communication bi-weekly",
        "Draft plan for InsureTech Solutions, been with us since 2023, annual revenue $25M, focus on cross-selling and digital transformation, weekly touchpoints, plan strategy session for April 2025",
        "Account planning for MegaBank Inc, $100M revenue, customer since Q3 2022, targeting 30% growth, schedule executive review on February 20 2025, monthly communication cadence"
    ]
    
    # Run benchmark
    benchmark_results = extractor.benchmark_performance(test_messages)
    
    # Show detailed example
    print("\n🎯 DETAILED EXTRACTION EXAMPLE:")
    print("-" * 60)
    result = extractor.extract_planning_data(test_messages[0])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("-" * 60)
    for key, value in benchmark_results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

