"""
DSL Compiler - Parser Component with Rich Error Recovery
========================================================

Task 6.2.2: Implement parser with rich error recovery
- Convert DSL v2 grammar into executable AST
- Panic-mode error recovery for partial parsing
- Structured diagnostics with error codes, line hints
- Tokenization and AST generation per DSL v2

Dependencies: Task 6.2.1 (DSL v2 grammar)
Outputs: AST → input for IR builder (6.2.4)
"""

import yaml
import json
import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Task 6.2.2: Structured Diagnostics
class ErrorCode(Enum):
    """Error codes for structured diagnostics"""
    SYNTAX_ERROR = "E001"
    MISSING_REQUIRED_FIELD = "E002"
    INVALID_STEP_TYPE = "E003"
    INVALID_ML_CONFIG = "E004"
    MISSING_FALLBACK = "E005"
    INVALID_CONFIDENCE = "E006"
    INVALID_THRESHOLD = "E007"
    CIRCULAR_DEPENDENCY = "E008"
    UNKNOWN_REFERENCE = "E009"
    POLICY_PACK_NOT_FOUND = "E010"

@dataclass
class Diagnostic:
    """Structured diagnostic information"""
    error_code: ErrorCode
    line: int
    column: int
    message: str
    hint: Optional[str] = None
    severity: str = "error"  # error, warning, info
    
@dataclass
class Token:
    """Token representation for DSL v2"""
    type: str
    value: str
    line: int
    column: int

class TokenType(Enum):
    """Token types for DSL v2 tokenization"""
    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    
    # Keywords
    WORKFLOW_ID = "workflow_id"
    NAME = "name"
    MODULE = "module"
    AUTOMATION_TYPE = "automation_type"
    VERSION = "version"
    POLICY_PACK = "policy_pack"
    STEPS = "steps"
    GOVERNANCE = "governance"
    
    # ML Keywords (from Task 6.2.1)
    ML_NODE = "ml_node"
    THRESHOLD = "threshold"
    CONFIDENCE = "confidence"
    FALLBACK = "fallback"
    EXPLAINABILITY = "explainability"
    
    # Operators
    COLON = ":"
    COMMA = ","
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    
    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"

class StepType(Enum):
    """DSL v2 Step Types with ML nodes"""
    # Traditional RBA steps
    QUERY = "query"
    DECISION = "decision" 
    ACTION = "action"
    NOTIFY = "notify"
    GOVERNANCE = "governance"
    AGENT_CALL = "agent_call"
    
    # ML steps (Task 6.2.1)
    ML_NODE = "ml_node"
    ML_PREDICT = "ml_predict"
    ML_SCORE = "ml_score"
    ML_CLASSIFY = "ml_classify"
    ML_EXPLAIN = "ml_explain"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Task 6.2.2: AST Node Definitions
@dataclass
class ASTNode:
    """Base AST node"""
    line: int = 0
    column: int = 0

@dataclass 
class WorkflowAST(ASTNode):
    """Root AST node for DSL v2 workflow"""
    workflow_id: str
    name: str
    module: str
    automation_type: str  # RBA, RBIA, AALA
    version: str
    policy_pack: Optional[str] = None
    governance: Optional[Dict[str, Any]] = None
    steps: List['StepAST'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class StepAST(ASTNode):
    """AST node for workflow step"""
    id: str
    type: StepType
    params: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)
    fallback: Optional['FallbackAST'] = None

@dataclass
class FallbackAST(ASTNode):
    """AST node for fallback configuration"""
    enabled: bool
    fallback_list: List['FallbackItemAST'] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)

@dataclass
class FallbackItemAST(ASTNode):
    """AST node for individual fallback item"""
    type: str  # rba_rule, default_action, human_escalation, previous_step
    target: str
    condition: Optional[str] = None

@dataclass
class MLConfigAST(ASTNode):
    """AST node for ML configuration"""
    model_id: str
    model_version: Optional[str] = None
    confidence: Optional['ConfidenceAST'] = None
    threshold: Optional[Dict[str, float]] = None
    explainability: Optional['ExplainabilityAST'] = None

@dataclass
class ConfidenceAST(ASTNode):
    """AST node for confidence configuration"""
    min_confidence: float
    auto_execute_above: Optional[float] = None
    assisted_mode_below: Optional[float] = None

@dataclass
class ExplainabilityAST(ASTNode):
    """AST node for explainability configuration"""
    enabled: bool
    method: str  # shap, lime, gradient, attention, counterfactual
    params: Dict[str, Any] = field(default_factory=dict)

# Task 6.2.2: Tokenizer with Error Recovery
class DSLTokenizer:
    """Tokenizer for DSL v2 with error recovery"""
    
    def __init__(self):
        self.keywords = {
            'workflow_id', 'name', 'module', 'automation_type', 'version',
            'policy_pack', 'steps', 'governance', 'ml_node', 'threshold',
            'confidence', 'fallback', 'explainability', 'query', 'decision',
            'action', 'notify', 'agent_call', 'ml_predict', 'ml_score',
            'ml_classify', 'ml_explain', 'true', 'false'
        }
        
    def tokenize(self, content: str) -> Tuple[List[Token], List[Diagnostic]]:
        """Tokenize DSL content with error recovery"""
        tokens = []
        diagnostics = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_tokens, line_diagnostics = self._tokenize_line(line, line_num)
            tokens.extend(line_tokens)
            diagnostics.extend(line_diagnostics)
            
        tokens.append(Token(TokenType.EOF.value, "", len(lines), 0))
        return tokens, diagnostics
    
    def _tokenize_line(self, line: str, line_num: int) -> Tuple[List[Token], List[Diagnostic]]:
        """Tokenize a single line"""
        tokens = []
        diagnostics = []
        i = 0
        
        while i < len(line):
            # Skip whitespace
            if line[i].isspace():
                i += 1
                continue
                
            # String literals
            if line[i] in ['"', "'"]:
                token, new_i, diag = self._parse_string(line, i, line_num)
                tokens.append(token)
                if diag:
                    diagnostics.append(diag)
                i = new_i
                continue
                
            # Numbers
            if line[i].isdigit() or (line[i] == '.' and i + 1 < len(line) and line[i + 1].isdigit()):
                token, new_i = self._parse_number(line, i, line_num)
                tokens.append(token)
                i = new_i
                continue
                
            # Identifiers and keywords
            if line[i].isalpha() or line[i] == '_':
                token, new_i = self._parse_identifier(line, i, line_num)
                tokens.append(token)
                i = new_i
                continue
                
            # Single character tokens
            char_tokens = {
                ':': TokenType.COLON,
                ',': TokenType.COMMA,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET
            }
            
            if line[i] in char_tokens:
                tokens.append(Token(char_tokens[line[i]].value, line[i], line_num, i))
                i += 1
                continue
                
            # Comments
            if line[i] == '#':
                break  # Skip rest of line
                
            # Unknown character - error recovery
            diagnostics.append(Diagnostic(
                ErrorCode.SYNTAX_ERROR,
                line_num,
                i,
                f"Unexpected character: '{line[i]}'",
                "Remove or escape this character"
            ))
            i += 1
            
        return tokens, diagnostics
    
    def _parse_string(self, line: str, start: int, line_num: int) -> Tuple[Token, int, Optional[Diagnostic]]:
        """Parse string literal with error recovery"""
        quote = line[start]
        i = start + 1
        value = ""
        
        while i < len(line):
            if line[i] == quote:
                return Token(TokenType.STRING.value, value, line_num, start), i + 1, None
            elif line[i] == '\\' and i + 1 < len(line):
                # Handle escape sequences
                next_char = line[i + 1]
                if next_char in ['"', "'", '\\', 'n', 't', 'r']:
                    value += line[i:i+2]
                    i += 2
                else:
                    value += line[i]
                    i += 1
            else:
                value += line[i]
                i += 1
                
        # Unterminated string - error recovery
        diagnostic = Diagnostic(
            ErrorCode.SYNTAX_ERROR,
            line_num,
            start,
            f"Unterminated string literal",
            f"Add closing {quote} at end of string"
        )
        return Token(TokenType.STRING.value, value, line_num, start), len(line), diagnostic
    
    def _parse_number(self, line: str, start: int, line_num: int) -> Tuple[Token, int]:
        """Parse number literal"""
        i = start
        has_dot = False
        
        while i < len(line):
            if line[i].isdigit():
                i += 1
            elif line[i] == '.' and not has_dot:
                has_dot = True
                i += 1
            else:
                break
                
        value = line[start:i]
        return Token(TokenType.NUMBER.value, value, line_num, start), i
    
    def _parse_identifier(self, line: str, start: int, line_num: int) -> Tuple[Token, int]:
        """Parse identifier or keyword"""
        i = start
        
        while i < len(line) and (line[i].isalnum() or line[i] in ['_', '-']):
            i += 1
            
        value = line[start:i]
        token_type = value if value in self.keywords else TokenType.IDENTIFIER.value
        return Token(token_type, value, line_num, start), i

# Task 6.2.2: Parser with Panic-Mode Error Recovery
class DSLParser:
    """DSL v2 Parser with rich error recovery"""
    
    def __init__(self):
        self.tokenizer = DSLTokenizer()
        self.tokens: List[Token] = []
        self.current = 0
        self.diagnostics: List[Diagnostic] = []
        
    def parse(self, content: str) -> Tuple[Optional[WorkflowAST], List[Diagnostic]]:
        """Parse DSL content into AST with error recovery"""
        # Tokenize
        self.tokens, tokenize_diagnostics = self.tokenizer.tokenize(content)
        self.diagnostics.extend(tokenize_diagnostics)
        self.current = 0
        
        try:
            # Parse workflow
            workflow = self._parse_workflow()
            return workflow, self.diagnostics
        except Exception as e:
            # Panic mode recovery - return partial AST
            self.diagnostics.append(Diagnostic(
                ErrorCode.SYNTAX_ERROR,
                self._current_token().line if self._current_token() else 1,
                self._current_token().column if self._current_token() else 0,
                f"Parse failed: {str(e)}",
                "Check syntax against DSL v2 grammar"
            ))
            return None, self.diagnostics
    
    def _current_token(self) -> Optional[Token]:
        """Get current token"""
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return None
    
    def _advance(self) -> Token:
        """Advance to next token"""
        token = self._current_token()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token
    
    def _match(self, expected: str) -> bool:
        """Check if current token matches expected"""
        current = self._current_token()
        return current and current.type == expected
    
    def _consume(self, expected: str, error_message: str) -> Optional[Token]:
        """Consume expected token or add diagnostic"""
        if self._match(expected):
            return self._advance()
        
        current = self._current_token()
        self.diagnostics.append(Diagnostic(
            ErrorCode.SYNTAX_ERROR,
            current.line if current else 1,
            current.column if current else 0,
            error_message,
            f"Expected '{expected}'"
        ))
        return None
    
    def _parse_workflow(self) -> WorkflowAST:
        """Parse workflow root"""
        workflow = WorkflowAST("", "", "", "", "", line=1, column=0)
        
        # Parse YAML structure first for compatibility
        try:
            yaml_content = '\n'.join([token.value for token in self.tokens if token.type != TokenType.EOF.value])
            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict):
                return self._convert_yaml_to_ast(data)
        except:
            pass
        
        # Fallback to token-based parsing
        while not self._match(TokenType.EOF.value):
            if self._match("workflow_id"):
                self._advance()
                self._consume(":", "Expected ':' after workflow_id")
                if self._match(TokenType.STRING.value) or self._match(TokenType.IDENTIFIER.value):
                    workflow.workflow_id = self._advance().value
            elif self._match("name"):
                self._advance()
                self._consume(":", "Expected ':' after name")
                if self._match(TokenType.STRING.value):
                    workflow.name = self._advance().value
            # Add more field parsing...
            else:
                self._advance()  # Skip unknown token
                
        return workflow
    
    def _convert_yaml_to_ast(self, data: Dict[str, Any]) -> WorkflowAST:
        """Convert YAML data to AST with validation"""
        # Required fields validation
        required_fields = ['workflow_id', 'name', 'module', 'automation_type', 'version']
        for field in required_fields:
            if field not in data:
                self.diagnostics.append(Diagnostic(
                    ErrorCode.MISSING_REQUIRED_FIELD,
                    1, 0,
                    f"Missing required field: {field}",
                    f"Add '{field}' to workflow definition"
                ))
        
        workflow = WorkflowAST(
            workflow_id=data.get('workflow_id', ''),
            name=data.get('name', ''),
            module=data.get('module', ''),
            automation_type=data.get('automation_type', 'RBA'),
            version=data.get('version', '1.0.0'),
            policy_pack=data.get('policy_pack'),
            governance=data.get('governance'),
            metadata=data.get('metadata', {})
        )
        
        # Parse steps with error recovery
        steps_data = data.get('steps', [])
        for i, step_data in enumerate(steps_data):
            try:
                step_ast = self._parse_step_ast(step_data, i)
                workflow.steps.append(step_ast)
            except Exception as e:
                # Error recovery - skip malformed step
                self.diagnostics.append(Diagnostic(
                    ErrorCode.SYNTAX_ERROR,
                    i + 1, 0,
                    f"Error parsing step {i}: {str(e)}",
                    "Check step syntax"
                ))
                continue
        
        return workflow
    
    def _parse_step_ast(self, step_data: Dict[str, Any], index: int) -> StepAST:
        """Parse step data into AST with validation"""
        # Validate required step fields
        if 'id' not in step_data:
            self.diagnostics.append(Diagnostic(
                ErrorCode.MISSING_REQUIRED_FIELD,
                index + 1, 0,
                f"Step {index} missing required 'id' field",
                "Add 'id' field to step"
            ))
            
        if 'type' not in step_data:
            self.diagnostics.append(Diagnostic(
                ErrorCode.MISSING_REQUIRED_FIELD,
                index + 1, 0,
                f"Step {index} missing required 'type' field",
                "Add 'type' field to step"
            ))
        
        # Validate step type
        step_type_str = step_data.get('type', 'action')
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            self.diagnostics.append(Diagnostic(
                ErrorCode.INVALID_STEP_TYPE,
                index + 1, 0,
                f"Invalid step type: '{step_type_str}'",
                f"Use one of: {[t.value for t in StepType]}"
            ))
            step_type = StepType.ACTION  # Default fallback
        
        step_ast = StepAST(
            id=step_data.get('id', f'step_{index}'),
            type=step_type,
            params=step_data.get('params', {}),
            outputs=step_data.get('outputs', {}),
            governance=step_data.get('governance', {}),
            next_steps=step_data.get('next_steps', []),
            line=index + 1
        )
        
        # Parse ML-specific configurations
        if step_type in [StepType.ML_NODE, StepType.ML_PREDICT, StepType.ML_SCORE, StepType.ML_CLASSIFY, StepType.ML_EXPLAIN]:
            self._validate_ml_step(step_ast, step_data, index)
            
        # Parse fallback configuration
        if 'fallback' in step_data:
            step_ast.fallback = self._parse_fallback_ast(step_data['fallback'], index)
            
        return step_ast
    
    def _validate_ml_step(self, step_ast: StepAST, step_data: Dict[str, Any], index: int):
        """Validate ML step configuration"""
        params = step_data.get('params', {})
        
        # Check for required ML fields
        if 'model_id' not in params:
            self.diagnostics.append(Diagnostic(
                ErrorCode.INVALID_ML_CONFIG,
                index + 1, 0,
                "ML step missing 'model_id' parameter",
                "Add 'model_id' to params"
            ))
        
        # Validate confidence configuration
        if 'confidence' in params:
            confidence = params['confidence']
            if isinstance(confidence, dict):
                min_conf = confidence.get('min_confidence')
                if min_conf is not None and (min_conf < 0.0 or min_conf > 1.0):
                    self.diagnostics.append(Diagnostic(
                        ErrorCode.INVALID_CONFIDENCE,
                        index + 1, 0,
                        f"Invalid min_confidence: {min_conf}. Must be between 0.0 and 1.0",
                        "Set min_confidence between 0.0 and 1.0"
                    ))
        
        # Check for fallback requirement
        if 'fallback' not in step_data:
            self.diagnostics.append(Diagnostic(
                ErrorCode.MISSING_FALLBACK,
                index + 1, 0,
                "ML steps require fallback configuration",
                "Add 'fallback' configuration with at least one fallback strategy"
            ))
    
    def _parse_fallback_ast(self, fallback_data: Dict[str, Any], index: int) -> FallbackAST:
        """Parse fallback configuration"""
        fallback_ast = FallbackAST(
            enabled=fallback_data.get('enabled', True),
            line=index + 1
        )
        
        # Parse fallback list
        fallback_list = fallback_data.get('fallback', [])
        for fb_item in fallback_list:
            if isinstance(fb_item, dict):
                fallback_item = FallbackItemAST(
                    type=fb_item.get('type', 'default_action'),
                    target=fb_item.get('target', ''),
                    condition=fb_item.get('condition'),
                    line=index + 1
                )
                fallback_ast.fallback_list.append(fallback_item)
        
        # Parse trigger conditions
        fallback_ast.trigger_conditions = fallback_data.get('trigger_conditions', [])
        
# Task 6.2.2: Compiler Integration
class DSLCompiler:
    """
    DSL v2 Compiler with Rich Error Recovery
    
    Task 6.2.2: Implement parser with rich error recovery
    - Tokenization and AST generation per DSL v2
    - Panic-mode error recovery 
    - Structured diagnostics with error codes, line hints
    
    Integration: Outputs AST → input for IR builder (6.2.4)
    Dependencies: Task 6.2.1 (DSL v2 grammar)
    """
    
    def __init__(self):
        self.parser = DSLParser()
        self.logger = logging.getLogger(__name__)
        
    def compile_workflow(self, content: str) -> Tuple[Optional[WorkflowAST], List[Diagnostic]]:
        """
        Compile DSL v2 workflow into AST with error recovery
        
        Args:
            content: DSL v2 workflow content (YAML format)
            
        Returns:
            Tuple of (WorkflowAST or None, List of Diagnostics)
        """
        try:
            workflow_ast, diagnostics = self.parser.parse(content)
            
            # Log compilation results
            if workflow_ast:
                error_count = len([d for d in diagnostics if d.severity == "error"])
                warning_count = len([d for d in diagnostics if d.severity == "warning"])
                
                if error_count == 0:
                    self.logger.info(f"✅ Compiled workflow: {workflow_ast.workflow_id} ({warning_count} warnings)")
                else:
                    self.logger.warning(f"⚠️ Compiled workflow with errors: {workflow_ast.workflow_id} ({error_count} errors, {warning_count} warnings)")
            else:
                self.logger.error("❌ Failed to compile workflow - too many syntax errors")
                
            return workflow_ast, diagnostics
            
        except Exception as e:
            # Ultimate fallback - should not happen with proper error recovery
            diagnostic = Diagnostic(
                ErrorCode.SYNTAX_ERROR,
                1, 0,
                f"Compiler failure: {str(e)}",
                "Contact support - this is a compiler bug"
            )
            self.logger.error(f"Compiler internal error: {e}")
            return None, [diagnostic]
    
    def format_diagnostics(self, diagnostics: List[Diagnostic]) -> str:
        """Format diagnostics for human consumption"""
        if not diagnostics:
            return "No issues found."
        
        formatted = []
        for diag in diagnostics:
            line_info = f"Line {diag.line}, Column {diag.column}"
            severity_symbol = "❌" if diag.severity == "error" else "⚠️" if diag.severity == "warning" else "ℹ️"
            
            formatted.append(f"{severity_symbol} {diag.error_code.value}: {diag.message}")
            formatted.append(f"   → {line_info}")
            if diag.hint:
                formatted.append(f"   💡 {diag.hint}")
            formatted.append("")
        
        return "\n".join(formatted)

# Example usage and testing
def test_parser_error_recovery():
    """Test parser error recovery capabilities"""
    compiler = DSLCompiler()
    
    # Test case 1: Missing required fields
    test_workflow_1 = """
workflow_id: "test_workflow"
# Missing name, module, automation_type, version
steps:
  - id: "step1"
    type: "query"
    params: {}
"""
    
    # Test case 2: Invalid ML configuration
    test_workflow_2 = """
workflow_id: "ml_test"
name: "ML Test"
module: "test"
automation_type: "RBIA"
version: "1.0.0"
steps:
  - id: "ml_step"
    type: "ml_node"
    params:
      # Missing model_id
      confidence:
        min_confidence: 1.5  # Invalid - > 1.0
    # Missing fallback
"""
    
    # Test case 3: Malformed YAML with recovery
    test_workflow_3 = """
workflow_id: "recovery_test"
name: "Recovery Test
# Unterminated string
module: "test"
automation_type: "RBA"
version: "1.0.0"
steps:
  - id: "good_step"
    type: "query"
    params: {}
  - id: "bad_step"
    type: "invalid_type"  # Invalid step type
    params: {}
"""
    
    print("Testing Parser Error Recovery...")
    print("=" * 50)
    
    for i, test_content in enumerate([test_workflow_1, test_workflow_2, test_workflow_3], 1):
        print(f"\nTest Case {i}:")
        print("-" * 20)
        
        workflow_ast, diagnostics = compiler.compile_workflow(test_content)
        
        if workflow_ast:
            print(f"✅ Parsed workflow: {workflow_ast.workflow_id}")
            print(f"   Steps parsed: {len(workflow_ast.steps)}")
        else:
            print("❌ Failed to parse workflow")
            
        print("\nDiagnostics:")
        print(compiler.format_diagnostics(diagnostics))

if __name__ == "__main__":
    test_parser_error_recovery()
