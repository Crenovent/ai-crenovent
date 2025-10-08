# 🧪 Testing Guide: Optimized Hierarchy Processing

This guide shows you how to test the **Super Smart RBA Agent** and verify the performance improvements (30 users in <5 seconds vs 10 minutes with LLM).

## 🚀 Quick Start Testing

### 1. **Simple Performance Test**
```bash
# Run the quick test script
cd ai-crenovent
python test_hierarchy_performance.py
```

This will test:
- ✅ Super Smart RBA Mapper performance
- ✅ Optimized Universal Mapper 
- ✅ Complete workflow (30 users target)
- ✅ Performance benchmark

### 2. **Full Test Suite**
```bash
# Run comprehensive tests
pytest tests/test_optimized_hierarchy_processing.py -v
```

## 📊 Expected Results

### **Performance Targets**
| Test | Target | Expected Result |
|------|--------|----------------|
| 30 users | <5 seconds | ✅ Should pass |
| 50 users | <8 seconds | ✅ Should pass |
| 100 users | <15 seconds | ✅ Should pass |
| Confidence | >90% | ✅ Should pass |

### **Sample Output**
```
🧠 Testing Super Smart RBA Mapper...
   📊 Input: 30 users with columns: ['Name', 'Email', 'Job Title', 'Manager Email', 'Department', 'Location']
   ✅ Processed in 0.234s
   🎯 Confidence: 94.2%
   🔍 Detected system: generic_hrms
   📈 Throughput: 128.2 users/sec

🚀 Testing Complete Optimized Workflow...
   ⏱️  Total processing time: 2.45s
   📊 Status: completed
   ✅ SUCCESS: 30 users processed in 2.45s
   🎯 PERFORMANCE TARGET MET: <5 seconds ✅
```

## 🧪 Test Categories

### **1. Component Tests**

#### **Super Smart RBA Mapper**
```python
from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
import pandas as pd

# Test basic mapping
mapper = SuperSmartRBAMapper()
df = pd.read_csv('test_data/sample_hierarchy_30_users.csv')

mapped_df, confidence, detected_system = mapper.map_csv_intelligently(df, tenant_id=1300)
print(f"Confidence: {confidence:.1%}, System: {detected_system}")
```

#### **Optimized Universal Mapper**
```python
from hierarchy_processor.core.optimized_universal_mapper import OptimizedUniversalMapper

# Test vectorized processing
mapper = OptimizedUniversalMapper(enable_caching=True)
df = pd.read_csv('test_data/sample_salesforce_format.csv')

mapped_df = mapper.map_any_hrms_to_crenovent_vectorized(df)
print(f"Processed {len(mapped_df)} records")
```

### **2. Workflow Tests**

#### **Complete Workflow Test**
```python
import asyncio
from src.rba.hierarchy_workflow_executor import process_csv_hierarchy_optimized

async def test_workflow():
    result = await process_csv_hierarchy_optimized(
        csv_file_path="test_data/sample_hierarchy_30_users.csv",
        tenant_id=1300,
        uploaded_by_user_id=1323
    )
    print(f"Status: {result.status}")

asyncio.run(test_workflow())
```

### **3. Performance Tests**

#### **Benchmark Different Sizes**
```python
from hierarchy_processor.core.super_smart_rba_mapper import SuperSmartRBAMapper
import time

mapper = SuperSmartRBAMapper()
sizes = [10, 30, 50, 100, 200]

for size in sizes:
    # Create test data
    df = create_test_csv(size)  # Your test data function
    
    start_time = time.time()
    mapped_df, confidence, system = mapper.map_csv_intelligently(df)
    processing_time = time.time() - start_time
    
    throughput = size / processing_time
    print(f"{size:3d} users: {processing_time:.3f}s ({throughput:.1f} users/sec)")
```

## 📁 Test Data Files

### **Available Test Files**
- `test_data/sample_hierarchy_30_users.csv` - Standard format (30 users)
- `test_data/sample_salesforce_format.csv` - Salesforce format (15 users)  
- `test_data/sample_messy_format.csv` - Messy column names (15 users)

### **Create Custom Test Data**
```python
import pandas as pd

# Create your own test CSV
data = [
    {'Name': 'John CEO', 'Email': 'john@company.com', 'Job Title': 'CEO', 'Manager Email': ''},
    {'Name': 'Jane VP', 'Email': 'jane@company.com', 'Job Title': 'VP Sales', 'Manager Email': 'john@company.com'},
    # Add more rows...
]

df = pd.DataFrame(data)
df.to_csv('my_test_data.csv', index=False)
```

## 🔍 Testing Different CSV Formats

### **1. Standard Format**
```csv
Name,Email,Job Title,Manager Email,Department,Location
John Smith,john@company.com,CEO,,Executive,New York
```

### **2. Salesforce Format**
```csv
Full Name,Email Address,Job Title,Reports To Email,Business Unit,Office Location
John Smith,john@company.com,CEO,,Executive,New York
```

### **3. Workday Format**
```csv
Worker Name,Work Email,Position Title,Manager Work Email,Organization Unit,Work Location
John Smith,john@company.com,CEO,,Executive,New York
```

### **4. Messy Format**
```csv
employee_full_name,work_email_address,current_job_title,direct_supervisor_email,dept_name,office_loc
John Smith,john@company.com,CEO,,EXEC,NYC
```

## ⚡ Performance Comparison Testing

### **Test Optimized vs Legacy**
```python
import time
from src.rba.hierarchy_workflow_executor import process_csv_hierarchy_via_rba

# Test optimized version
start = time.time()
result_optimized = await process_csv_hierarchy_via_rba(
    csv_file_path="test.csv",
    tenant_id=1300,
    uploaded_by_user_id=1323,
    use_optimized=True  # Use optimized
)
optimized_time = time.time() - start

# Test legacy version (if available)
start = time.time()
result_legacy = await process_csv_hierarchy_via_rba(
    csv_file_path="test.csv", 
    tenant_id=1300,
    uploaded_by_user_id=1323,
    use_optimized=False  # Use legacy
)
legacy_time = time.time() - start

print(f"Optimized: {optimized_time:.2f}s")
print(f"Legacy: {legacy_time:.2f}s") 
print(f"Improvement: {legacy_time/optimized_time:.1f}x faster")
```

## 🐛 Debugging Tests

### **Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your tests with detailed logging
```

### **Check Confidence Scores**
```python
# If confidence is low, check why
mapper = SuperSmartRBAMapper()
df = pd.read_csv('problematic.csv')

mapped_df, confidence, system = mapper.map_csv_intelligently(df)

if confidence < 0.8:
    print(f"Low confidence: {confidence:.1%}")
    print(f"Columns: {list(df.columns)}")
    print(f"Detected system: {system}")
    
    # Check mapping stats
    stats = mapper.get_mapping_stats()
    print(f"Mapping stats: {stats}")
```

### **Test Individual Components**
```python
# Test field detection only
context = mapper._analyze_csv_context(df)
field_mappings = mapper._intelligent_field_detection(df.columns.tolist(), context)

print(f"Context: {context}")
print(f"Field mappings: {field_mappings}")
```

## 📈 Monitoring Performance

### **Track Key Metrics**
- **Processing Time**: Should be <5s for 30 users
- **Confidence Score**: Should be >90% for standard formats
- **Throughput**: Should be >6 users/second
- **Memory Usage**: Should be <512MB
- **LLM Calls**: Should be 0 (zero!)

### **Performance Assertions**
```python
import time

start_time = time.time()
result = await process_csv_hierarchy_optimized(csv_path, 1300, 1323)
processing_time = time.time() - start_time

# Key assertions
assert processing_time < 5.0, f"Too slow: {processing_time:.2f}s"
assert result.status == "completed", f"Failed: {result.error}"

# Check for LLM usage (should be zero)
if hasattr(result, 'final_output'):
    llm_calls = result.final_output.get('llm_calls_made', 0)
    assert llm_calls == 0, f"LLM calls detected: {llm_calls}"
```

## 🎯 Success Criteria

### **✅ Tests Should Pass If:**
1. **30 users process in <5 seconds**
2. **Confidence score >90% for standard formats**
3. **Zero LLM API calls made**
4. **All required fields mapped correctly**
5. **Memory usage <512MB**
6. **No errors or exceptions**

### **⚠️ Investigate If:**
1. Processing time >5 seconds for 30 users
2. Confidence score <80%
3. Missing required fields (Name, Email)
4. Errors in workflow execution
5. High memory usage

## 🔧 Troubleshooting

### **Common Issues**

#### **ImportError: Module not found**
```bash
# Make sure you're in the right directory
cd ai-crenovent

# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### **Low Confidence Scores**
- Check CSV column names
- Verify data quality
- Test with sample files first

#### **Performance Issues**
- Check system resources
- Verify caching is enabled
- Test with smaller datasets first

#### **Workflow Failures**
- Check file paths
- Verify tenant_id and user_id
- Check database connectivity

## 📞 Getting Help

If tests fail or performance is poor:

1. **Check the logs** for detailed error messages
2. **Run individual component tests** to isolate issues
3. **Compare with sample data** to verify format compatibility
4. **Check system resources** (CPU, memory)
5. **Verify all dependencies** are installed

The optimized system should dramatically outperform the legacy LLM-based approach while maintaining high accuracy!
