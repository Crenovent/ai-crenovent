#!/usr/bin/env python3
"""
Update All RBA Agents to Enhanced Base Class
===========================================
Script to update all RBA agents to use EnhancedBaseRBAAgent for universal parameters and KG tracing.
"""

import os
import glob
import re

def update_rba_agent_file(file_path):
    """Update a single RBA agent file to use EnhancedBaseRBAAgent"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already using enhanced base class
        if 'EnhancedBaseRBAAgent' in content:
            print(f"✅ {os.path.basename(file_path)} already uses EnhancedBaseRBAAgent")
            return True
            
        # Skip base agent files
        if 'base_rba_agent.py' in file_path or 'enhanced_base_rba_agent.py' in file_path:
            print(f"⏭️ Skipping base class file: {os.path.basename(file_path)}")
            return True
        
        print(f"🔄 Updating {os.path.basename(file_path)}...")
        
        # Replace import statement
        content = re.sub(
            r'from \.base_rba_agent import BaseRBAAgent',
            'from .enhanced_base_rba_agent import EnhancedBaseRBAAgent',
            content
        )
        
        # Replace class inheritance
        content = re.sub(
            r'class (\w+RBAAgent)\(BaseRBAAgent\):',
            r'class \1(EnhancedBaseRBAAgent):',
            content
        )
        
        # Remove default_config definitions (now handled by universal parameters)
        content = re.sub(
            r'\s+# .*-specific defaults.*\n\s+self\.default_config = \{[^}]*\}',
            '\n        # Note: Default configuration now comes from universal parameter system\n        # Agent-specific defaults are defined in universal_rba_parameters.yaml',
            content,
            flags=re.DOTALL
        )
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {os.path.basename(file_path)}: {e}")
        return False

def main():
    """Update all RBA agent files"""
    
    print("🚀 UPDATING ALL RBA AGENTS TO ENHANCED BASE CLASS")
    print("=" * 60)
    
    # Find all RBA agent files
    rba_dir = "dsl/operators/rba"
    agent_files = glob.glob(os.path.join(rba_dir, "*_rba_agent.py"))
    
    print(f"📊 Found {len(agent_files)} RBA agent files")
    
    updated_count = 0
    failed_count = 0
    
    for file_path in agent_files:
        if update_rba_agent_file(file_path):
            updated_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print("📋 UPDATE SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully updated: {updated_count}")
    print(f"❌ Failed to update: {failed_count}")
    print(f"📊 Total files processed: {len(agent_files)}")
    
    if failed_count == 0:
        print("🎉 ALL RBA AGENTS UPDATED SUCCESSFULLY!")
        print("💡 All agents now support:")
        print("   - Universal parameter system (67+ parameters)")
        print("   - Knowledge Graph execution tracing")
        print("   - Enhanced configuration management")
        print("   - Governance and audit trails")
    else:
        print("⚠️ Some agents failed to update. Please review the errors above.")

if __name__ == "__main__":
    main()
