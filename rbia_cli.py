#!/usr/bin/env python3
"""
RBIA CLI - Task 6.2.68
======================

CLI for local dev (rbia compile ...)
- Mirrors API functionality
- Local development tool
- Command-line interface for compiler
"""

import argparse
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import compiler services
try:
    from dsl.compiler.parser import DSLCompiler
    from dsl.compiler.ir import ASTToIRConverter
    from dsl.compiler.plan_manifest_generator import PlanManifestGenerator
    from dsl.compiler.plan_hash_service import PlanHashService
    from dsl.compiler.digital_signature_service import DigitalSignatureService
    from dsl.compiler.sandbox_mode_service import SandboxModeService, SandboxLevel
    from dsl.compiler.cost_annotation_service import CostAnnotationService
    from dsl.compiler.plan_labels_service import PlanLabelsService
except ImportError:
    # Fallback for when running outside full environment
    pass

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_file(file_path: str) -> Dict[str, Any]:
    """Load YAML or JSON file"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    try:
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                print(f"Error: Unsupported file format {path.suffix}")
                sys.exit(1)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def save_file(data: Any, file_path: str, format: str = 'json'):
    """Save data to file"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2, default=str)
        print(f"Output saved to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        sys.exit(1)

def cmd_compile(args):
    """Compile DSL workflow to IR and manifest"""
    print(f"Compiling {args.input_file}...")
    
    # Load input file
    workflow_data = load_file(args.input_file)
    
    try:
        # Parse workflow
        compiler = DSLCompiler()
        ast = compiler.compile_workflow_from_dict(workflow_data)
        
        # Convert to IR
        converter = ASTToIRConverter()
        ir_data = converter.convert_ast_to_ir(ast)
        
        # Apply services if requested
        if args.sandbox:
            sandbox_service = SandboxModeService()
            sandbox_level = SandboxLevel(args.sandbox_level) if args.sandbox_level else SandboxLevel.BASIC
            ir_data = sandbox_service.enable_sandbox_mode(ir_data, sandbox_level)
        
        if args.cost_analysis:
            cost_service = CostAnnotationService()
            ir_data = cost_service.annotate_plan_with_costs(ir_data)
        
        if args.labels:
            labels_service = PlanLabelsService()
            ir_data = labels_service.analyze_and_label_plan(ir_data)
        
        # Generate manifest
        manifest_generator = PlanManifestGenerator()
        manifest = manifest_generator.generate_plan_manifest(ir_data)
        
        # Generate hash if requested
        if args.hash:
            hash_service = PlanHashService()
            plan_hash = hash_service.compute_manifest_hash(manifest)
            manifest['plan_hash'] = plan_hash
            print(f"Plan hash: {plan_hash}")
        
        # Save output
        output_file = args.output or f"{Path(args.input_file).stem}_compiled.json"
        save_file(manifest, output_file, args.format)
        
        print(f"✅ Compilation successful")
        print(f"   Nodes: {len(ir_data.get('nodes', []))}")
        print(f"   Edges: {len(ir_data.get('edges', []))}")
        
        if args.sandbox:
            print(f"   Sandbox mode: {args.sandbox_level or 'basic'}")
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def cmd_validate(args):
    """Validate DSL workflow"""
    print(f"Validating {args.input_file}...")
    
    workflow_data = load_file(args.input_file)
    
    try:
        compiler = DSLCompiler()
        ast = compiler.compile_workflow_from_dict(workflow_data)
        
        # Basic validation
        issues = []
        
        # Check required fields
        if 'workflow_id' not in workflow_data:
            issues.append("Missing required field: workflow_id")
        
        if 'steps' not in workflow_data:
            issues.append("Missing required field: steps")
        
        # Check steps
        steps = workflow_data.get('steps', [])
        for i, step in enumerate(steps):
            if 'id' not in step:
                issues.append(f"Step {i}: Missing required field: id")
            if 'type' not in step:
                issues.append(f"Step {i}: Missing required field: type")
        
        if issues:
            print("❌ Validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print("✅ Validation passed")
            print(f"   Workflow ID: {workflow_data.get('workflow_id')}")
            print(f"   Steps: {len(steps)}")
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def cmd_hash(args):
    """Generate plan hash"""
    print(f"Generating hash for {args.input_file}...")
    
    data = load_file(args.input_file)
    
    try:
        hash_service = PlanHashService()
        plan_hash = hash_service.compute_manifest_hash(data)
        
        print(f"Plan hash: {plan_hash}")
        
        if args.output:
            hash_data = {
                'input_file': args.input_file,
                'plan_hash': plan_hash,
                'algorithm': 'sha256',
                'generated_at': hash_service._get_current_timestamp()
            }
            save_file(hash_data, args.output)
    
    except Exception as e:
        print(f"❌ Hash generation failed: {e}")
        sys.exit(1)

def cmd_sandbox(args):
    """Enable/disable sandbox mode"""
    if args.action == 'enable':
        print(f"Enabling sandbox mode for {args.input_file}...")
        
        ir_data = load_file(args.input_file)
        
        try:
            sandbox_service = SandboxModeService()
            sandbox_level = SandboxLevel(args.level) if args.level else SandboxLevel.BASIC
            
            ir_data = sandbox_service.enable_sandbox_mode(ir_data, sandbox_level)
            
            output_file = args.output or f"{Path(args.input_file).stem}_sandbox.json"
            save_file(ir_data, output_file)
            
            print(f"✅ Sandbox mode enabled ({sandbox_level.value})")
        
        except Exception as e:
            print(f"❌ Sandbox enable failed: {e}")
            sys.exit(1)
    
    elif args.action == 'disable':
        print(f"Disabling sandbox mode for {args.input_file}...")
        
        ir_data = load_file(args.input_file)
        
        try:
            sandbox_service = SandboxModeService()
            ir_data = sandbox_service.disable_sandbox_mode(ir_data)
            
            output_file = args.output or f"{Path(args.input_file).stem}_production.json"
            save_file(ir_data, output_file)
            
            print("✅ Sandbox mode disabled")
        
        except Exception as e:
            print(f"❌ Sandbox disable failed: {e}")
            sys.exit(1)
    
    elif args.action == 'status':
        print(f"Checking sandbox status for {args.input_file}...")
        
        ir_data = load_file(args.input_file)
        
        try:
            sandbox_service = SandboxModeService()
            status = sandbox_service.get_sandbox_status(ir_data)
            
            print(f"Sandbox enabled: {'Yes' if status['sandbox_enabled'] else 'No'}")
            if status['sandbox_enabled']:
                print(f"Sandbox level: {status['sandbox_level']}")
                print(f"Restrictions: {', '.join(status.get('restrictions', []))}")
        
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            sys.exit(1)

def cmd_cost(args):
    """Analyze plan costs"""
    print(f"Analyzing costs for {args.input_file}...")
    
    ir_data = load_file(args.input_file)
    
    try:
        cost_service = CostAnnotationService()
        annotated_ir = cost_service.annotate_plan_with_costs(ir_data)
        
        cost_summary = annotated_ir.get('metadata', {}).get('cost_summary', {})
        
        print("✅ Cost analysis complete:")
        print(f"   Monthly estimate: ${cost_summary.get('total_monthly_estimate_usd', 0):.2f}")
        print(f"   Cost per execution: ${cost_summary.get('total_cost_per_execution_usd', 0):.4f}")
        
        cost_breakdown = cost_summary.get('cost_breakdown_monthly', {})
        if cost_breakdown:
            print("   Cost breakdown:")
            for category, cost in cost_breakdown.items():
                if cost > 0:
                    print(f"     {category}: ${cost:.2f}")
        
        if args.output:
            dashboard_data = cost_service.generate_finops_dashboard_data(annotated_ir)
            save_file(dashboard_data, args.output)
    
    except Exception as e:
        print(f"❌ Cost analysis failed: {e}")
        sys.exit(1)

def cmd_version(args):
    """Show version information"""
    version_info = {
        'rbia_cli': '1.0.0',
        'compiler_version': '6.2.0',
        'python_version': sys.version,
        'supported_formats': ['yaml', 'yml', 'json']
    }
    
    if args.format == 'json':
        print(json.dumps(version_info, indent=2))
    else:
        print("RBIA CLI")
        print(f"Version: {version_info['rbia_cli']}")
        print(f"Compiler: {version_info['compiler_version']}")
        print(f"Python: {version_info['python_version']}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='RBIA CLI - Local development tool for RBIA workflows',
        prog='rbia'
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Output format')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile DSL workflow')
    compile_parser.add_argument('input_file', help='Input DSL file')
    compile_parser.add_argument('--output', '-o', help='Output file')
    compile_parser.add_argument('--hash', action='store_true', help='Generate plan hash')
    compile_parser.add_argument('--sandbox', action='store_true', help='Enable sandbox mode')
    compile_parser.add_argument('--sandbox-level', choices=['basic', 'strict', 'development'], help='Sandbox level')
    compile_parser.add_argument('--cost-analysis', action='store_true', help='Include cost analysis')
    compile_parser.add_argument('--labels', action='store_true', help='Generate plan labels')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate DSL workflow')
    validate_parser.add_argument('input_file', help='Input DSL file')
    
    # Hash command
    hash_parser = subparsers.add_parser('hash', help='Generate plan hash')
    hash_parser.add_argument('input_file', help='Input file')
    hash_parser.add_argument('--output', '-o', help='Output file for hash data')
    
    # Sandbox command
    sandbox_parser = subparsers.add_parser('sandbox', help='Manage sandbox mode')
    sandbox_parser.add_argument('action', choices=['enable', 'disable', 'status'], help='Sandbox action')
    sandbox_parser.add_argument('input_file', help='Input IR file')
    sandbox_parser.add_argument('--level', choices=['basic', 'strict', 'development'], help='Sandbox level')
    sandbox_parser.add_argument('--output', '-o', help='Output file')
    
    # Cost command
    cost_parser = subparsers.add_parser('cost', help='Analyze plan costs')
    cost_parser.add_argument('input_file', help='Input IR file')
    cost_parser.add_argument('--output', '-o', help='Output file for cost data')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    # Route to command handlers
    if args.command == 'compile':
        cmd_compile(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'hash':
        cmd_hash(args)
    elif args.command == 'sandbox':
        cmd_sandbox(args)
    elif args.command == 'cost':
        cmd_cost(args)
    elif args.command == 'version':
        cmd_version(args)

if __name__ == '__main__':
    main()
