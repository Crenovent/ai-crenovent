# RBIA Release Process - Version Matrix
# Task 6.5.80: Version matrix (DSL↔compiler↔runtime), changelogs

## Version Compatibility Matrix

| DSL Version | Compiler Version | Runtime Version | Status | Release Date |
|-------------|------------------|-----------------|--------|--------------|
| 2.0.0       | 2.0.0           | 2.0.0          | Current| 2024-01-15   |
| 1.9.0       | 1.9.0           | 1.9.0          | Supported| 2023-12-01 |
| 1.8.0       | 1.8.0           | 1.8.0          | Supported| 2023-11-01 |
| 1.7.0       | 1.7.0           | 1.7.0          | EOL    | 2023-10-01   |

## Compatibility Rules

### Forward Compatibility
- **DSL:** Version N works with Compiler N+1 (with warnings)
- **Compiler:** Version N works with Runtime N+1 (degraded features)
- **Runtime:** Version N works with DSL N-1 (legacy support)

### Breaking Changes
- **Major Version:** May introduce breaking changes
- **Minor Version:** Backward compatible, new features
- **Patch Version:** Bug fixes only, fully compatible

## Release Schedule

### Weekly Cadence
- **Monday:** Feature freeze for current sprint
- **Tuesday:** Integration testing and validation
- **Wednesday:** Security and compliance review
- **Thursday:** Release candidate builds
- **Friday:** Production deployment (if approved)

### Release Types
- **Hotfix:** Emergency patches (any day)
- **Minor:** Feature releases (weekly)
- **Major:** Breaking changes (quarterly)

## Component Dependencies

```yaml
dsl_grammar:
  current_version: "2.0.0"
  compatible_compilers: ["2.0.0", "1.9.0"]
  breaking_changes: ["match/when syntax", "feature refs"]

compiler:
  current_version: "2.0.0"
  compatible_dsl: ["2.0.0", "1.9.0", "1.8.0"]
  compatible_runtime: ["2.0.0", "1.9.0"]
  new_features: ["predicate pushdown", "ML coalescing"]

runtime:
  current_version: "2.0.0"
  compatible_compiler: ["2.0.0", "1.9.0"]
  supported_dsl: ["2.0.0", "1.9.0", "1.8.0"]
  deprecated_features: ["legacy fallback syntax"]
```

## Migration Paths

### DSL 1.x → 2.0 Migration
1. **Assessment:** Run compatibility checker
2. **Grammar Update:** Update to new syntax
3. **Policy Binding:** Add governance metadata
4. **Validation:** Test with new compiler
5. **Deployment:** Gradual rollout

### Compiler 1.x → 2.0 Migration
1. **Backup:** Export current configurations
2. **Install:** Deploy new compiler version
3. **Validate:** Run regression tests
4. **Rollback Plan:** Keep previous version available
5. **Monitor:** Watch for compilation errors

## Changelog Template

### Version 2.0.0 (2024-01-15)

#### Added
- Match/when control flow with ML predicates
- Feature reference syntax (feature.namespace.name)
- Predicate pushdown optimizer
- ML node coalescing optimizer
- Conversational DSL scaffolder
- Policy snippet library
- Lint error taxonomy
- Training course materials

#### Changed
- Grammar extended with control flow constructs
- Type system enhanced for ML models
- Performance optimizations in compiler
- Enhanced error messages with categories

#### Deprecated
- Legacy fallback syntax (will be removed in 3.0)
- Old confidence threshold format
- Manual policy binding

#### Removed
- None (fully backward compatible)

#### Fixed
- Memory leaks in long-running compilations
- Race conditions in parallel optimization
- Incorrect type inference for vector types

#### Security
- Enhanced secret detection in DSL validation
- Improved tenant isolation in compilation
- Stricter policy enforcement

## Support Matrix

### Supported Versions
- **Current (2.0.x):** Full support, active development
- **Previous (1.9.x):** Security fixes only
- **Legacy (1.8.x):** Critical fixes only
- **EOL (≤1.7.x):** No support

### End-of-Life Policy
- **Notice Period:** 6 months before EOL
- **Migration Support:** Free migration assistance
- **Extended Support:** Available for enterprise customers

## Release Validation Checklist

### Pre-Release
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Security scan clean
- [ ] Performance benchmarks within limits
- [ ] Documentation updated
- [ ] Changelog complete

### Post-Release
- [ ] Deployment successful
- [ ] Monitoring alerts configured
- [ ] Support team notified
- [ ] Customer communication sent
- [ ] Rollback plan tested
