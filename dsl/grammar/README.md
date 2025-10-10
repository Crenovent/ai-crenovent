# RBIA DSL v2 Grammar Specification
# Task 6.2.1: Define DSL v2 grammar with ml_node, threshold, confidence, policy_pack, fallback[], explainability

## Overview

This document defines the RBIA DSL v2 grammar that extends the traditional Rule-Based Automation (RBA) syntax with Machine Learning decision nodes, creating a hybrid workflow language for the RBIA (Rule-Based Intelligent Automation) system.

## Design Principles

1. **Backward Compatibility**: All existing RBA workflows continue to work unchanged
2. **Hybrid Support**: Seamlessly combine deterministic rules with ML decision nodes
3. **Governance-First**: Policy packs and compliance constraints embedded at compile time
4. **Fallback Mandatory**: Every ML node must have degradation paths (RBIA→RBA)
5. **Explainability Built-In**: ML nodes require explainability configuration
6. **Type Safety**: Strong typing for ML model inputs/outputs and governance constraints

## Grammar Format

The grammar is defined in Extended Backus-Naur Form (EBNF) as specified in Task 6.2.1.

## Key New Constructs

### 1. ml_node Step Type
```ebnf
ml_step = "ml_node" | "ml_predict" | "ml_score" | "ml_classify" | "ml_explain" ;
```

The `ml_node` is the unified ML step type that can handle any ML primitive (predict, score, classify, explain) based on the model configuration.

### 2. policy_pack Workflow Attribute
```ebnf
"policy_pack" ":" policy_pack?
```

References a governance policy collection that gets bound at compile time, enabling governance-by-design.

### 3. threshold Configuration
```ebnf
threshold_param = "threshold" ":" number 
                | "confidence_threshold" ":" number
                | "score_threshold" ":" number ;
```

Flexible threshold configuration for different ML primitives.

### 4. confidence Block
```ebnf
confidence_param = "confidence" ":" "{" 
                  "min_confidence" ":" number
                  "auto_execute_above" ":" number?
                  "assisted_mode_below" ":" number?
                  "}" ;
```

Defines confidence-based execution policies:
- `min_confidence`: minimum confidence to proceed
- `auto_execute_above`: confidence level for automatic execution
- `assisted_mode_below`: confidence level requiring human assistance

### 5. fallback[] Array Syntax
```ebnf
fallback_block = "{" fallback_config "}" ;
fallback_config = "enabled" ":" boolean
                 "fallback" ":" "[" fallback_list "]"
                 "trigger_conditions" ":" trigger_conditions? ;
```

Mandatory fallback configuration for ML nodes with multiple fallback strategies.

### 6. explainability Configuration
```ebnf
explainability_param = "explainability" ":" "{" 
                      "enabled" ":" boolean
                      "method" ":" explainability_method
                      "params" ":" explainability_params?
                      "}" ;
```

Built-in explainability support with multiple methods (SHAP, LIME, gradient, attention, counterfactual).

## Compilation Semantics

### Type Checking
- ML node inputs must conform to model contracts
- Confidence values must be between 0.0 and 1.0
- Policy pack references must resolve to valid governance policies

### Fallback Validation
- Every ML node must have at least one fallback strategy
- Fallback targets must exist in the workflow
- Circular fallback dependencies are detected and rejected

### Governance Binding
- Policy packs are resolved and bound at compile time
- Residency constraints are validated against model deployment regions
- SLA budgets are accumulated across the workflow execution path

## Migration Path

### RBA → RBIA Migration
1. **Phase 1**: Add `policy_pack` to existing RBA workflows
2. **Phase 2**: Replace critical decision points with `ml_node` steps
3. **Phase 3**: Add `fallback` configurations pointing to original RBA rules
4. **Phase 4**: Enable `explainability` for transparency

### Backward Compatibility Guarantee
- All DSL v1 syntax remains valid in DSL v2
- `automation_type: "RBA"` workflows ignore ML-specific constructs
- Gradual adoption without breaking changes

## Implementation Notes

This grammar specification enables:
- **Parser Generation**: Can be used with ANTLR, PEG.js, or Rust pest
- **IDE Support**: Language Server Protocol (LSP) implementation for the Builder
- **Static Analysis**: Type checking, cycle detection, policy validation
- **Code Generation**: Compilation to executable workflow plans

## Related Tasks

This grammar specification supports the following downstream tasks:
- **6.2.2**: Parser implementation with rich error recovery
- **6.2.3**: Language Server (LSP) for Builder IDE support
- **6.2.4**: Intermediate Representation (IR) for hybrid graphs
- **6.2.5**: Type system implementation
- **6.2.6**: Schema linking to ontology & model contracts
