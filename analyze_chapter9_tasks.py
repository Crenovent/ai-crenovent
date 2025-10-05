#!/usr/bin/env python3
"""
Chapter 9 Task Analysis Script
Extracts all tasks from Chapter 9 and checks implementation status in codebase
"""

import re
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set

def extract_chapter9_tasks(orchestrator_file: str) -> Dict[str, List[Dict[str, str]]]:
    """Extract all Chapter 9 tasks from orchestrator.txt"""
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find Chapter 9 sections
    chapter9_sections = {
        '9.1': 'Planes',
        '9.2': 'Core Components', 
        '9.3': 'End-to-End Flow',
        '9.4': 'Multi-Tenant Enforcement',
        '9.5': 'Resilience & Fallback'
    }
    
    all_tasks = {}
    
    for section_id, section_name in chapter9_sections.items():
        print(f"🔍 Extracting tasks from Chapter {section_id}: {section_name}")
        
        # Pattern to match task lines like "9.1.1	Task description	Outcome	Tech	Notes"
        task_pattern = rf'^{re.escape(section_id)}\.(\d+)\s+(.+?)\s+(.+?)\s+(.+?)\s+(.+?)$'
        
        tasks = []
        lines = content.split('\n')
        
        for line in lines:
            match = re.match(task_pattern, line, re.MULTILINE)
            if match:
                task_num = match.group(1)
                task_id = f"{section_id}.{task_num}"
                description = match.group(2).strip()
                outcome = match.group(3).strip()
                technology = match.group(4).strip()
                notes = match.group(5).strip()
                
                tasks.append({
                    'task_id': task_id,
                    'description': description,
                    'outcome': outcome,
                    'technology': technology,
                    'notes': notes
                })
        
        all_tasks[section_id] = tasks
        print(f"   ✅ Found {len(tasks)} tasks in section {section_id}")
    
    return all_tasks

def check_task_implementation(task: Dict[str, str], codebase_path: str) -> Dict[str, any]:
    """Check if a task is implemented in the codebase"""
    
    task_id = task['task_id']
    description = task['description'].lower()
    
    # Keywords to search for based on task description
    search_terms = []
    
    # Extract key terms from description
    if 'orchestrator' in description or 'orchestration' in description:
        search_terms.extend(['orchestrator', 'routing', 'dispatcher'])
    if 'registry' in description:
        search_terms.extend(['registry', 'capability'])
    if 'policy' in description:
        search_terms.extend(['policy', 'governance'])
    if 'evidence' in description:
        search_terms.extend(['evidence', 'audit'])
    if 'override' in description:
        search_terms.extend(['override', 'ledger'])
    if 'sla' in description:
        search_terms.extend(['sla', 'error_budget'])
    if 'tenant' in description:
        search_terms.extend(['tenant', 'multi_tenant'])
    if 'lineage' in description:
        search_terms.extend(['lineage', 'traceability'])
    if 'dashboard' in description:
        search_terms.extend(['dashboard', 'grafana'])
    if 'finops' in description:
        search_terms.extend(['finops', 'cost'])
    if 'consent' in description:
        search_terms.extend(['consent', 'gdpr'])
    if 'trust' in description:
        search_terms.extend(['trust_scor', 'reliability'])
    if 'chaos' in description or 'resilience' in description:
        search_terms.extend(['chaos', 'resilience', 'fallback'])
    if 'retry' in description:
        search_terms.extend(['retry', 'backoff'])
    if 'anomaly' in description:
        search_terms.extend(['anomaly', 'detection'])
    
    # Default search terms if none found
    if not search_terms:
        # Extract first few meaningful words
        words = description.split()[:3]
        search_terms = [word.strip('.,()[]') for word in words if len(word) > 3]
    
    # Search for implementations using Python file search
    found_files = []
    implementation_score = 0
    
    # Search in ai-crenovent directory
    ai_crenovent_path = os.path.join(codebase_path, 'ai-crenovent')
    
    for term in search_terms[:5]:  # Limit to first 5 terms
        try:
            # Search in Python files using os.walk
            for root, dirs, files in os.walk(ai_crenovent_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read().lower()
                                if term.lower() in content:
                                    found_files.append(file_path)
                                    implementation_score += 1
                        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
                            continue
        except Exception:
            continue
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    # Determine implementation status
    if implementation_score >= 10:
        status = "IMPLEMENTED"
    elif implementation_score >= 3:
        status = "PARTIALLY_IMPLEMENTED"
    else:
        status = "NOT_IMPLEMENTED"
    
    return {
        'task_id': task_id,
        'status': status,
        'implementation_score': implementation_score,
        'found_files': found_files[:10],  # Limit to first 10 files
        'search_terms': search_terms[:5]
    }

def categorize_tasks(tasks: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize tasks by type"""
    
    categories = {
        'Infrastructure & Architecture': [],
        'Governance & Compliance': [],
        'Data & Analytics': [],
        'Security & Access Control': [],
        'Monitoring & Observability': [],
        'DevOps & Operations': [],
        'API & Integration': [],
        'Testing & Quality': [],
        'Documentation': []
    }
    
    for task in tasks:
        desc = task['description'].lower()
        
        if any(term in desc for term in ['build', 'deploy', 'configure', 'implement', 'automate']):
            if any(term in desc for term in ['policy', 'governance', 'compliance', 'evidence', 'override', 'ledger']):
                categories['Governance & Compliance'].append(task)
            elif any(term in desc for term in ['dashboard', 'monitor', 'observability', 'telemetry', 'sla']):
                categories['Monitoring & Observability'].append(task)
            elif any(term in desc for term in ['data', 'lineage', 'schema', 'quality']):
                categories['Data & Analytics'].append(task)
            elif any(term in desc for term in ['security', 'rbac', 'consent', 'encryption', 'key']):
                categories['Security & Access Control'].append(task)
            elif any(term in desc for term in ['test', 'chaos', 'validation']):
                categories['Testing & Quality'].append(task)
            elif any(term in desc for term in ['api', 'integration', 'connector']):
                categories['API & Integration'].append(task)
            elif any(term in desc for term in ['runbook', 'ops', 'devops']):
                categories['DevOps & Operations'].append(task)
            else:
                categories['Infrastructure & Architecture'].append(task)
        elif any(term in desc for term in ['define', 'document', 'publish']):
            categories['Documentation'].append(task)
        else:
            categories['Infrastructure & Architecture'].append(task)
    
    return categories

def generate_report(all_tasks: Dict[str, List[Dict[str, str]]], 
                   all_implementations: Dict[str, List[Dict[str, any]]]) -> str:
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("# Chapter 9 Task Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    total_tasks = sum(len(tasks) for tasks in all_tasks.values())
    total_implemented = 0
    total_partial = 0
    total_not_implemented = 0
    
    # Count implementation status
    for section_implementations in all_implementations.values():
        for impl in section_implementations:
            if impl['status'] == 'IMPLEMENTED':
                total_implemented += 1
            elif impl['status'] == 'PARTIALLY_IMPLEMENTED':
                total_partial += 1
            else:
                total_not_implemented += 1
    
    report.append(f"## Executive Summary")
    report.append(f"- **Total Tasks**: {total_tasks}")
    report.append(f"- **Implemented**: {total_implemented} ({total_implemented/total_tasks*100:.1f}%)")
    report.append(f"- **Partially Implemented**: {total_partial} ({total_partial/total_tasks*100:.1f}%)")
    report.append(f"- **Not Implemented**: {total_not_implemented} ({total_not_implemented/total_tasks*100:.1f}%)")
    report.append("")
    
    # Section-by-section analysis
    for section_id, tasks in all_tasks.items():
        implementations = all_implementations[section_id]
        
        section_implemented = sum(1 for impl in implementations if impl['status'] == 'IMPLEMENTED')
        section_partial = sum(1 for impl in implementations if impl['status'] == 'PARTIALLY_IMPLEMENTED')
        section_not_implemented = sum(1 for impl in implementations if impl['status'] == 'NOT_IMPLEMENTED')
        
        report.append(f"## Chapter {section_id} Analysis")
        report.append(f"- **Total Tasks**: {len(tasks)}")
        report.append(f"- **Implemented**: {section_implemented}")
        report.append(f"- **Partially Implemented**: {section_partial}")
        report.append(f"- **Not Implemented**: {section_not_implemented}")
        report.append("")
        
        # List implemented tasks
        if section_implemented > 0:
            report.append(f"### ✅ Implemented Tasks ({section_implemented})")
            for impl in implementations:
                if impl['status'] == 'IMPLEMENTED':
                    task = next(t for t in tasks if t['task_id'] == impl['task_id'])
                    report.append(f"- **{impl['task_id']}**: {task['description']}")
            report.append("")
        
        # List partially implemented tasks
        if section_partial > 0:
            report.append(f"### ⚠️ Partially Implemented Tasks ({section_partial})")
            for impl in implementations:
                if impl['status'] == 'PARTIALLY_IMPLEMENTED':
                    task = next(t for t in tasks if t['task_id'] == impl['task_id'])
                    report.append(f"- **{impl['task_id']}**: {task['description']}")
                    if impl['found_files']:
                        report.append(f"  - Found in: {', '.join(impl['found_files'][:3])}")
            report.append("")
        
        # List not implemented tasks
        if section_not_implemented > 0:
            report.append(f"### ❌ Not Implemented Tasks ({section_not_implemented})")
            for impl in implementations:
                if impl['status'] == 'NOT_IMPLEMENTED':
                    task = next(t for t in tasks if t['task_id'] == impl['task_id'])
                    report.append(f"- **{impl['task_id']}**: {task['description']}")
            report.append("")
    
    # Task categorization
    all_flat_tasks = []
    for tasks in all_tasks.values():
        all_flat_tasks.extend(tasks)
    
    categories = categorize_tasks(all_flat_tasks)
    
    report.append("## Task Categories")
    for category, tasks in categories.items():
        if tasks:
            report.append(f"### {category} ({len(tasks)} tasks)")
            for task in tasks[:5]:  # Show first 5 tasks
                report.append(f"- {task['task_id']}: {task['description'][:80]}...")
            if len(tasks) > 5:
                report.append(f"- ... and {len(tasks) - 5} more tasks")
            report.append("")
    
    return "\n".join(report)

def main():
    """Main execution function"""
    
    print("🚀 Starting Chapter 9 Task Analysis...")
    
    # Paths
    orchestrator_file = "docs for context/orchestrator.txt"
    codebase_path = "."
    
    if not os.path.exists(orchestrator_file):
        print(f"❌ Error: {orchestrator_file} not found")
        return
    
    # Extract tasks
    print("\n📋 Extracting Chapter 9 tasks...")
    all_tasks = extract_chapter9_tasks(orchestrator_file)
    
    total_tasks = sum(len(tasks) for tasks in all_tasks.values())
    print(f"✅ Total tasks extracted: {total_tasks}")
    
    # Check implementations
    print("\n🔍 Checking task implementations...")
    all_implementations = {}
    
    for section_id, tasks in all_tasks.items():
        print(f"   Analyzing section {section_id}...")
        implementations = []
        
        for task in tasks:
            impl = check_task_implementation(task, codebase_path)
            implementations.append(impl)
        
        all_implementations[section_id] = implementations
    
    # Generate report
    print("\n📊 Generating analysis report...")
    report = generate_report(all_tasks, all_implementations)
    
    # Save report
    report_file = "CHAPTER_9_ANALYSIS_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Analysis complete! Report saved to {report_file}")
    
    # Print summary
    total_implemented = sum(
        sum(1 for impl in implementations if impl['status'] == 'IMPLEMENTED')
        for implementations in all_implementations.values()
    )
    total_partial = sum(
        sum(1 for impl in implementations if impl['status'] == 'PARTIALLY_IMPLEMENTED')
        for implementations in all_implementations.values()
    )
    total_not_implemented = sum(
        sum(1 for impl in implementations if impl['status'] == 'NOT_IMPLEMENTED')
        for implementations in all_implementations.values()
    )
    
    print(f"\n📈 Summary:")
    print(f"   Total Tasks: {total_tasks}")
    print(f"   Implemented: {total_implemented} ({total_implemented/total_tasks*100:.1f}%)")
    print(f"   Partially Implemented: {total_partial} ({total_partial/total_tasks*100:.1f}%)")
    print(f"   Not Implemented: {total_not_implemented} ({total_not_implemented/total_tasks*100:.1f}%)")

if __name__ == "__main__":
    main()
