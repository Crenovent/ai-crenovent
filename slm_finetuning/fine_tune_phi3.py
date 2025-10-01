

#!/usr/bin/env python3
"""
Fine-tune Phi-3.5-mini for account planning form extraction
Uses QLoRA for efficient fine-tuning on consumer hardware
"""
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanningFormExtractorTrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # QLoRA configuration for efficient training
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA configuration"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with QLoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer loaded successfully")

    def format_training_data(self, examples: List[Dict]) -> List[str]:
        """Format training examples for instruction tuning"""
        formatted_examples = []
        
        for example in examples:
            # Create instruction-following format
            prompt = f"""<|system|>
You are an expert AI assistant specialized in extracting structured information from business account planning requests. Extract all relevant information and return it as a properly formatted JSON object.

<|user|>
{example['input']}

<|assistant|>
{json.dumps(example['output'], ensure_ascii=False)}"""
            
            formatted_examples.append(prompt)
        
        return formatted_examples

    def tokenize_function(self, examples):
        """Tokenize training examples"""
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=2048,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    def prepare_dataset(self, data_file: str):
        """Prepare dataset for training - supports both JSON and JSONL"""
        logger.info(f"Loading dataset from: {data_file}")
        
        # Detect file format and load appropriately
        if data_file.endswith('.jsonl'):
            # Load JSONL format (for large datasets)
            raw_data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0:
                        logger.info(f"   Loaded {line_num:,} examples...")
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
        else:
            # Load regular JSON format
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        logger.info(f"✅ Loaded {len(raw_data):,} training examples")
        
        # Format data
        formatted_texts = self.format_training_data(raw_data)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # Split into train/validation
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"✅ Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset

    def train(self, train_dataset, eval_dataset, output_dir: str = "phi3-planning-extractor"):
        """Fine-tune the model"""
        logger.info("🚀 Starting fine-tuning...")
        
        # Training arguments optimized for large datasets (100k+)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Fewer epochs for large datasets
            per_device_train_batch_size=2,  # Slightly larger batch size
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Larger accumulation for stability
            warmup_steps=1000,  # More warmup for large datasets
            learning_rate=3e-5,  # Slightly lower LR for stability
            fp16=True,
            logging_steps=100,  # Less frequent logging
            save_steps=2000,  # Less frequent saves
            eval_steps=2000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            max_steps=50000,  # Limit training steps for efficiency
            report_to=None,  # Disable wandb/tensorboard for speed
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Training complete! Model saved to: {output_dir}")
        return output_dir

    def test_model(self, model_dir: str, test_prompt: str):
        """Test the fine-tuned model"""
        logger.info("🧪 Testing fine-tuned model...")
        
        # Load fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Format test prompt
        formatted_prompt = f"""<|system|>
You are an expert AI assistant specialized in extracting structured information from business account planning requests. Extract all relevant information and return it as a properly formatted JSON object.

<|user|>
{test_prompt}

<|assistant|>
"""
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        logger.info("✅ Test completed!")
        return response

def main():
    # Initialize trainer
    trainer = PlanningFormExtractorTrainer()
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Auto-detect the latest training data file
    import glob
    possible_files = [
        "planning_training_data_100000.jsonl",
        "planning_training_data_*.jsonl", 
        "planning_training_data_*.json",
        "planning_training_data.json"
    ]
    
    data_file = None
    for pattern in possible_files:
        files = glob.glob(pattern)
        if files:
            # Get the largest file (most examples)
            data_file = max(files, key=lambda f: os.path.getsize(f))
            break
    
    if not data_file:
        logger.error("❌ No training data found!")
        logger.info("Please run: python data_generator.py 100000")
        return
    
    logger.info(f"🎯 Using training data: {data_file}")
    file_size = os.path.getsize(data_file) / (1024 * 1024)
    logger.info(f"📊 File size: {file_size:.1f} MB")
    
    train_dataset, eval_dataset = trainer.prepare_dataset(data_file)
    
    # Train model
    model_dir = trainer.train(train_dataset, eval_dataset)
    
    # Test model
    test_prompt = "Create account plan for TechCorp Insurance, customer since Q2 2024, revenue around 25M, highlighting renewal risks and cross-sell opportunities, schedule quarterly review on March 15 2025, communication weekly"
    
    result = trainer.test_model(model_dir, test_prompt)
    
    print("\n🎯 TEST RESULT:")
    print("-" * 50)
    print(result)

if __name__ == "__main__":
    main()
