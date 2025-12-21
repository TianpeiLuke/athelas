---
tags:
  - resource
  - best-practices
  - sop
  - refactoring
  - code-quality
  - testing
  - numerical-equivalence
keywords:
  - algorithm refactoring
  - numerical equivalence
  - functional equivalence
  - code refactoring
  - best practices
  - testing strategy
  - design patterns
  - code quality
topics:
  - software engineering
  - refactoring
  - quality assurance
  - algorithm implementation
  - testing methodology
language: python
date of note: 2025-12-19
---

# Algorithm-Preserving Refactoring: Standard Operating Procedure

## Purpose

This Standard Operating Procedure (SOP) provides a systematic framework for refactoring algorithm-heavy code while maintaining numerical and functional equivalence. It is designed to prevent production failures that result from prioritizing architectural improvements over algorithmic correctness.

**Key Principle**: In algorithm-heavy refactorings, **preserving exact behavior is the ONLY non-negotiable requirement**. Architecture, patterns, and optimizations are secondary and optional.

## Scope

This SOP applies to refactoring of:
- Machine learning algorithms and models
- Numerical computation code
- Mathematical optimization algorithms
- Signal processing and data transformations
- Financial calculations
- Scientific computing code
- Any code where numerical precision matters

## When to Use This SOP

✅ **USE THIS SOP when refactoring**:
- Machine learning training or inference code
- Loss functions, gradient computations, optimizers
- Data preprocessing or transformation pipelines
- Numerical algorithms (sorting, searching, optimization)
- Code with complex mathematical operations
- Code that depends on external numerical libraries
- Code that produces numerical outputs validated against benchmarks

❌ **This SOP may be overkill for**:
- Pure UI/frontend code with no numerical computation
- Simple CRUD operations
- Configuration-only changes
- Pure infrastructure code (unless it affects numerical behavior)

## Core Principles

### Principle 1: Correctness First, Everything Else Second

```
Priority Order (Non-Negotiable):
1. Numerical/Functional Equivalence (MANDATORY)
2. Maintains Production Stability (MANDATORY)
3. Clean Architecture (SECONDARY)
4. Design Patterns (SECONDARY)
5. Performance Optimization (OPTIONAL)
```

**Reality Check**: Good architecture ≠ Correct algorithms. You can have beautiful, clean code that produces wrong results.

### Principle 2: Separate Concerns

Refactoring and optimization are **distinct activities**:

```
Phase 1: Refactor (MUST complete first)
├─ Goal: Preserve exact behavior
├─ Constraint: Zero algorithmic changes
├─ Test: Numerical equivalence verified
└─ Output: Maintainable code with identical behavior

Phase 2: Optimize (ONLY after Phase 1)
├─ Goal: Improve performance
├─ Constraint: Must maintain equivalence
├─ Test: Profile → optimize bottlenecks → re-verify
└─ Output: Faster code with verified equivalent behavior
```

**Never mix these phases**. Each optimization must be verified independently.

### Principle 3: Test What Matters

Traditional test coverage is insufficient for algorithm refactoring:

```
❌ INSUFFICIENT TESTING:
✅ Does code compile?
✅ Does code run without crashing?
✅ Does training complete?
✅ Is test coverage >80%?

✅ REQUIRED TESTING:
✅ Do numerical outputs match legacy within tolerance?
✅ Do weights/parameters evolve identically?
✅ Does model actually learn (metrics improve)?
✅ Are there no NaN/Inf values?
✅ Does behavior match with edge cases?
```

### Principle 4: Understand Dependencies Completely

```
What to Document (MANDATORY):
├─ Python dependencies (requirements.txt)
├─ C/C++ libraries (.so files, headers)
├─ System dependencies (CUDA, MKL, etc.)
├─ Modified/forked libraries
├─ Data format requirements
│   ├─ Input shapes
│   ├─ Label shapes
│   └─ Output shapes
└─ API compatibility constraints
```

**Critical**: Syntactic similarity ≠ Functional equivalence. A custom fork with the same API may have fundamentally different behavior.

### Principle 5: Refactor in Layers

```
❌ WRONG: Change everything at once
✓ Architectural changes
✓ Algorithm changes
✓ Optimization changes
→ Cannot isolate root cause of failures

✅ CORRECT: One conceptual change per layer
Layer 1: Extract base class (no algorithm changes)
  → Verify: Numerical equivalence ✓
Layer 2: Add factory pattern (no algorithm changes)
  → Verify: Numerical equivalence ✓
Layer 3: Optimize if needed (OPTIONAL)
  → Verify: Numerical equivalence + performance ✓
```

### Principle 6: Production Data Characteristics Matter

```
❌ WRONG: Test with synthetic balanced data
# Test data: Random, balanced, no edge cases
X_test = np.random.randn(1000, 50)
y_test = np.random.randint(0, 2, size=(1000,))

✅ CORRECT: Test with production characteristics
# Production data: Imbalanced, hierarchical, edge cases
data = load_production_sample()
# - Extreme imbalance (110 vs 133,969 samples)
# - Hierarchical relationships
# - Constant predictions (early training)
# - Zero variance scenarios
```

### Principle 7: Optimize Later, or Never

```
The Optimization Decision Tree:

Is refactoring complete?
├─ NO → ❌ DO NOT optimize
└─ YES → Does equivalence test pass?
    ├─ NO → ❌ DO NOT optimize, fix bugs first
    └─ YES → Have you profiled?
        ├─ NO → ❌ DO NOT guess, profile first
        └─ YES → Is this code a bottleneck?
            ├─ NO → ❌ DO NOT optimize, it's fast enough
            └─ YES → Is improvement >20%?
                ├─ NO → ⚠️ Probably not worth it
                └─ YES → ✅ OK to optimize IF equivalence preserved
```

**Remember**: Correct slow code can be optimized. Incorrect fast code is just wrong.

---

## Pre-Refactoring Phase

### Step 1: Establish Baseline

**Objective**: Document current behavior as ground truth.

**Actions** (MANDATORY):

1. **Document System Behavior**
   ```python
   # Capture baseline outputs
   baseline_outputs = {
       'predictions': legacy_model.predict(X_test),
       'gradients': legacy_loss.compute_gradients(X_train, y_train),
       'weights': legacy_optimizer.get_weights(),
       'metrics': legacy_model.evaluate(X_val, y_val)
   }
   
   # Save to disk for comparison
   np.save('baseline_outputs.npy', baseline_outputs)
   ```

2. **Document Data Shapes**
   ```markdown
   ## Data Format Analysis (COMPLETE BEFORE CODING)
   
   ### Legacy Data Shapes
   - Input shape: X.shape = (n_samples, n_features) = (133969, 47)
   - Label shape: y.shape = (n_samples, n_tasks) = (133969, 6)
   - Output shape: predictions.shape = (n_samples, n_tasks) = (133969, 6)
   - Intermediate shapes: [Document all reshape operations]
   
   ### Library Requirements
   - Library: lightgbmmt (CUSTOM FORK, not standard lightgbm)
   - Input requirements: Accepts multi-column labels
   - Label requirements: shape (n_samples, n_tasks)
   - Output requirements: shape (n_samples, n_tasks)
   - Documentation: [URL to library docs]
   
   ### Compatibility Analysis
   - Input compatible? ✓ Both accept (n_samples, n_features)
   - Labels compatible? ❌ Standard LightGBM requires 1D, we have 2D
   - Output compatible? ❌ Standard returns 1D, we need 2D
   - Decision: ✅ Must use custom fork OR redesign approach
   ```

3. **Identify All Dependencies**
   ```bash
   # Complete dependency scan
   
   # Python imports
   grep -r "^import\|^from" --include="*.py" > python_dependencies.txt
   
   # C/C++ libraries
   find . -name "*.so" -o -name "*.dylib" > compiled_libraries.txt
   
   # Custom C extensions
   find . -name "*.cpp" -o -name "*.c" -o -name "*.h" > cpp_code.txt
   
   # Compare with standard packages
   pip show lightgbm > standard_lightgbm.txt
   ls -R lightgbmmt/ > custom_lightgbmmt.txt
   
   # Document differences
   diff standard_lightgbm.txt custom_lightgbmmt.txt > library_differences.txt
   ```

4. **Document Edge Cases**
   ```python
   # Collect edge cases from production
   edge_cases = {
       'constant_predictions': X[predictions_constant],
       'zero_variance': X[variance == 0],
       'extreme_imbalance': X[task_sizes < 100],
       'missing_values': X[contains_nan],
       'hierarchical_constraints': X[subtask_implies_main]
   }
   
   # Save for testing
   save_edge_cases('edge_cases.pkl', edge_cases)
   ```

5. **Measure Performance Baseline**
   ```python
   import time
   
   # Measure current performance
   start = time.time()
   result = legacy_model.train(X_train, y_train)
   baseline_time = time.time() - start
   
   # Document baseline
   baseline_metrics = {
       'training_time': baseline_time,
       'memory_peak': get_peak_memory(),
       'throughput': len(X_train) / baseline_time
   }
   ```

**Deliverables**:
- [ ] Baseline outputs saved to disk
- [ ] Data format documentation completed
- [ ] All dependencies documented
- [ ] Edge cases collected
- [ ] Performance baseline measured

**Gate**: Cannot proceed without completing all deliverables.

---

### Step 2: Verify Library Compatibility

**Objective**: Ensure target libraries support required data formats BEFORE writing any code.

**Actions** (MANDATORY):

1. **Read Library Documentation**
   ```python
   """
   Standard LightGBM Documentation Check:
   
   lightgbm.Dataset(data, label=None, ...)
   
   Parameters:
   - label: array-like of shape (n_samples,) or None
     Label of the data. Should be 1-dimensional array.
   
   CRITICAL: Our labels are shape (n_samples, n_tasks)
   → 2D array INCOMPATIBLE with standard library
   """
   ```

2. **Test With Standard Library**
   ```python
   # Validation test (DO NOT SKIP)
   import lightgbm as lgb  # Standard library
   
   try:
       # Attempt to create dataset with our data shapes
       dataset = lgb.Dataset(X_train, label=y_train)
       predictions = model.predict(X_test)
       
       # Verify shapes match requirements
       assert predictions.shape == expected_shape, \
           f"Shape mismatch: {predictions.shape} != {expected_shape}"
       
       print("✅ Standard library compatible")
       
   except Exception as e:
       print(f"❌ INCOMPATIBLE: {e}")
       print("→ Must use custom fork OR redesign approach")
       # Document why legacy works
       # Make go/no-go decision
   ```

3. **Document Compatibility Decision**
   ```markdown
   ## Library Compatibility Decision
   
   Date: 2025-12-19
   Decision: Use custom lightgbmmt fork
   
   ### Rationale
   - Standard LightGBM requires 1D labels: (n_samples,)
   - Our data has multi-task labels: (n_samples, n_tasks)
   - Standard library test failed with: "label must be 1-dimensional"
   - Legacy uses custom C++ modified fork: lightgbmmt
   
   ### Risk Assessment
   - ✅ Custom fork is maintained and tested
   - ✅ Fork supports our required data shapes
   - ⚠️ Fork must be kept in sync with upstream
   - ⚠️ Cannot migrate to standard library without redesign
   
   ### Alternative Considered
   - Redesign to use standard library (would require architecture change)
   - Estimated effort: 3-4 weeks
   - Risk: High (changes core architecture)
   - Decision: Use custom fork (lower risk, faster)
   ```

**Deliverables**:
- [ ] Library documentation reviewed
- [ ] Compatibility test executed
- [ ] Decision documented with rationale
- [ ] Risk assessment completed

**Gate**: Cannot start coding until compatibility verified and decision documented.

---

### Step 3: Create Numerical Regression Tests

**Objective**: Define tests that will verify equivalence throughout refactoring.

**Test Types** (MANDATORY):

1. **Output Equivalence Tests**
   ```python
   def test_predictions_match_legacy():
       """Verify predictions match within tolerance"""
       # Load baseline
       baseline = np.load('baseline_outputs.npy', allow_pickle=True).item()
       
       # Run refactored
       refactored_preds = refactored_model.predict(X_test)
       
       # Compare
       np.testing.assert_allclose(
           baseline['predictions'],
           refactored_preds,
           rtol=1e-6,
           err_msg="Predictions diverged from baseline"
       )
   ```

2. **Iteration-by-Iteration Tests**
   ```python
   def test_weight_evolution():
       """Verify weights evolve identically across iterations"""
       for iteration in range(100):
           # Legacy weights at iteration
           legacy_weights = legacy_loss.get_weights(iteration)
           
           # Refactored weights at iteration
           refactored_weights = refactored_loss.get_weights(iteration)
           
           # Must match exactly
           np.testing.assert_allclose(
               legacy_weights,
               refactored_weights,
               rtol=1e-6,
               err_msg=f"Weights diverged at iteration {iteration}"
           )
   ```

3. **Learning Verification Tests**
   ```python
   def test_model_actually_learns():
       """Verify model improves over iterations"""
       metrics = []
       
       for iteration in range(100):
           metric = refactored_model.evaluate(X_val, y_val)
           metrics.append(metric)
           
           # Check for learning (metrics should improve)
           if iteration > 10:
               recent = metrics[-10:]
               assert not all(m == recent[0] for m in recent), \
                   f"Metric frozen at {recent[0]} for 10 iterations - model not learning!"
   ```

4. **Edge Case Tests**
   ```python
   def test_constant_predictions():
       """Handle constant predictions without NaN/Inf"""
       # Create constant prediction scenario
       preds = np.ones(1000) * 0.5
       
       # Should not crash or produce NaN
       grad, hess = refactored_loss.objective(preds, train_data)
       
       assert not np.any(np.isnan(grad)), "Gradients contain NaN"
       assert not np.any(np.isnan(hess)), "Hessian contains NaN"
       assert not np.any(np.isinf(grad)), "Gradients contain Inf"
       assert not np.any(np.isinf(hess)), "Hessian contains Inf"
   ```

5. **Production-Scale Tests**
   ```python
   def test_with_production_characteristics():
       """Test with realistic data distribution"""
       # Load production sample
       prod_data = load_production_sample()
       
       # Test with extreme imbalance
       imbalanced = prod_data[prod_data.task_size < 100]
       result = refactored_model.train(imbalanced)
       assert result.converged, "Failed on imbalanced data"
       
       # Test with hierarchical constraints
       hierarchical = prod_data[prod_data.has_hierarchy]
       result = refactored_model.train(hierarchical)
       assert result.converged, "Failed on hierarchical data"
   ```

**Test Framework Template**:
```python
# tests/regression/test_numerical_equivalence.py

class NumericRegressionTest:
    """Framework for testing numerical equivalence"""
    
    def __init__(self, legacy_impl, refactored_impl):
        self.legacy = legacy_impl
        self.refactored = refactored_impl
        self.baseline = self.load_baseline()
    
    def load_baseline(self):
        """Load pre-computed baseline outputs"""
        return np.load('baseline_outputs.npy', allow_pickle=True).item()
    
    def test_outputs_match(self, inputs, tolerance=1e-6):
        """Verify outputs match within tolerance"""
        legacy_output = self.legacy.compute(inputs)
        refactored_output = self.refactored.compute(inputs)
        
        np.testing.assert_allclose(
            legacy_output,
            refactored_output,
            rtol=tolerance,
            err_msg="Outputs diverged beyond tolerance"
        )
    
    def test_all_edge_cases(self):
        """Test all documented edge cases"""
        edge_cases = load_edge_cases('edge_cases.pkl')
        
        for case_name, case_data in edge_cases.items():
            with self.subTest(case=case_name):
                self.test_outputs_match(case_data)
```

**Deliverables**:
- [ ] Output equivalence tests written
- [ ] Iteration-by-iteration tests written
- [ ] Learning verification tests written
- [ ] Edge case tests written
- [ ] Production-scale tests written
- [ ] All tests passing with legacy code

**Gate**: Cannot start refactoring until all tests pass with legacy code.

---

## Refactoring Phase

### Phase 1: Refactor for Correctness ONLY

**Objective**: Improve code organization while preserving exact numerical behavior.

**Rules** (NON-NEGOTIABLE):

1. **ZERO Optimizations**
   - No caching
   - No performance improvements
   - No "better" algorithms
   - No formula changes
   - No learning rate modifications

2. **Preserve EXACT Behavior**
   ```python
   # ✅ CORRECT: Keep exact legacy behavior
   def objective(self, preds, train_data):
       # Process predictions every time (no caching)
       preds_mat = expit(preds.reshape(...))
       
       # Keep legacy normalization (don't remove)
       grad_n = self.normalize(grad_i)
       
       # Use exact legacy values (don't "improve")
       learning_rate = 0.1  # Not 0.5, use legacy value
       
       return grad, hess
   
   # ❌ WRONG: Adding "optimization" during refactoring
   def objective(self, preds, train_data):
       # ❌ Don't add caching
       if id(preds) in self._cache:
           return self._cache[id(preds)]
       
       # ❌ Don't remove steps
       # grad_n = self.normalize(grad_i)  # Commented out!
       
       # ❌ Don't change values
       learning_rate = 0.5  # Changed from 0.1!
   ```

3. **Architectural Changes Only**
   - Extract base classes
   - Apply design patterns
   - Eliminate code duplication
   - Improve naming
   - Add documentation

**Refactoring Approach**:

```
Layer-by-Layer Refactoring:

Layer 1: Extract Base Class
├─ Extract shared methods to base class
├─ No algorithm changes
├─ Run tests: ✓ All pass
└─ Commit: "Extract BaseLossFunction (no behavior change)"

Layer 2: Apply Factory Pattern
├─ Add factory for object creation
├─ No algorithm changes
├─ Run tests: ✓ All pass
└─ Commit: "Add LossFactory (no behavior change)"

Layer 3: Extract Strategy Pattern
├─ Extract weight update strategies
├─ No algorithm changes
├─ Run tests: ✓ All pass
└─ Commit: "Extract WeightUpdateStrategy (no behavior change)"

[Continue one layer at a time]
```

**After EACH Layer**:
1. Run numerical equivalence tests
2. Verify all tests pass
3. Commit with clear message
4. If tests fail → revert and debug

**Example Commit Messages**:
```
✅ GOOD:
- "Extract BaseLossFunction (no behavior change)"
- "Add LossFactory (no behavior change)"
- "Rename confusing variable names (no behavior change)"

❌ BAD:
- "Refactored everything"
- "Improved loss function"
- "Optimized gradient computation"
```

**Verification Checklist** (After EVERY change):
```markdown
- [ ] All numerical equivalence tests pass
- [ ] No NaN/Inf values in outputs
- [ ] Iteration-by-iteration behavior identical
- [ ] Learning occurs (metrics improve)
- [ ] Edge cases handled correctly
- [ ] Performance within 10% of baseline
```

**Deliverables**:
- [ ] Code refactored in layers
- [ ] Each layer verified independently
- [ ] Clear commit history
- [ ] All tests passing
- [ ] No algorithmic changes

**Gate**: Phase 2 cannot start until ALL Phase 1 tests pass.

---

### Phase 2: Optimization (OPTIONAL)

**Objective**: Improve performance ONLY if needed and ONLY after correctness verified.

**Decision Tree**:

```
Should we optimize?

Q1: Is Phase 1 complete?
├─ NO → ❌ STOP: Finish refactoring first
└─ YES → Continue

Q2: Do all equivalence tests pass?
├─ NO → ❌ STOP: Fix bugs first
└─ YES → Continue

Q3: Is there a performance problem?
├─ NO → ✅ DONE: No optimization needed
└─ YES → Continue

Q4: Have you profiled to find bottlenecks?
├─ NO → ❌ STOP: Profile first, don't guess
└─ YES → Continue

Q5: Is this code actually a bottleneck?
├─ NO → ✅ DONE: Don't optimize, it's fast enough
└─ YES → Continue

Q6: Will optimization provide >20% improvement?
├─ NO → ⚠️ RECONSIDER: Risk may not be worth it
└─ YES → ✅ OK to optimize (with verification)
```

**Optimization Process**:

1. **Profile to Find Bottlenecks**
   ```python
   import cProfile
   import pstats
   
   # Profile actual bottlenecks
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Run training
   model.train(train_data)
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   
   # Example output:
   """
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    45.23    0.452    45.23    0.452 tree_builder.py:123(build_tree)  ← REAL bottleneck!
      100     2.31    0.023     2.31    0.023 loss.py:45(compute_gradients)
      100     0.15    0.001     0.15    0.001 loss.py:67(preprocess_predictions)  ← NOT a bottleneck!
   """
   ```

2. **Optimize ONLY Proven Bottlenecks**
   ```python
   # ✅ CORRECT: Optimize proven bottleneck
   # Profile shows tree_builder takes 94% of time
   def optimize_tree_building():
       # Add parallel tree building
       # Use more efficient data structures
       pass
   
   # ❌ WRONG: Optimize non-bottleneck
   # Profile shows preprocessing takes 0.3% of time
   def dont_optimize_this():
       # Don't add caching here!
       # It's already fast enough
       pass
   ```

3. **Verify After EACH Optimization**
   ```python
   def test_optimization_preserves_correctness():
       """Run after EACH optimization"""
       
       # 1. Numerical equivalence still holds
       np.testing.assert_allclose(
           baseline_output,
           optimized_output,
           rtol=1e-6
       )
       
       # 2. Performance actually improved
       baseline_time = benchmark(refactored_code)
       optimized_time = benchmark(optimized_code)
       
       improvement = (baseline_time - optimized_time) / baseline_time
       print(f"Performance improvement: {improvement*100:.1f}%")
       
       # 3. Was optimization worth it?
       if improvement < 0.05:  # Less than 5%
           print("⚠️ Optimization not worth complexity")
           print("Consider reverting to simpler code")
   ```

4. **Document Each Optimization**
   ```python
   """
   Optimization: Parallel tree building
   
   Profile Data:
   - Before: tree_builder = 45.2s (94% of total)
   - After: tree_builder = 22.1s (79% of total)
   - Improvement: 51% faster
   
   Verification:
   - Numerical equivalence: ✓ PASSED (rtol=1e-6)
   - Edge cases: ✓ PASSED (all 15 cases)
   - Learning: ✓ PASSED (AUC improves)
   
   Trade-offs:
   - Code complexity: +15% (added threading)
   - Memory usage: +10% (thread overhead)
   - Benefit: Worth it (51% speedup)
   """
   ```

**Key Principle**: Most refactorings need ZERO optimization.

```
Refactoring Goals:
✅ PRIMARY: Preserve exact behavior (correctness)
✅ SECONDARY: Improve code organization (maintainability)
❌ NOT A GOAL: Improve performance (optimization)

Optimization is:
- A SEPARATE activity
- ONLY after correctness verified
- ONLY for proven bottlenecks
- OPTIONAL (most code is fast enough)
```

**Deliverables** (If optimization performed):
- [ ] Profile data showing bottleneck
- [ ] Measured improvement (X% faster)
- [ ] Equivalence tests still passing
- [ ] Each optimization documented
- [ ] Trade-offs analyzed

---

## Integration Verification Phase

**Objective**: Ensure new infrastructure is actually used, not just created.

**The Problem**: Common failure pattern in code generation and refactoring:
1. Design new data structure or component
2. Implement methods and properties
3. **FAIL**: Never connect to actual system
4. Tests pass because infrastructure exists, but integration never verified

**Integration Checklist** (MANDATORY for any new infrastructure):

### Step 1: Design Phase
```markdown
- [ ] Document purpose of the component
- [ ] Define interface (methods, properties)
- [ ] Identify calling code that will use it
- [ ] Specify integration points
```

### Step 2: Implementation Phase
```markdown
- [ ] Implement component methods
- [ ] Add inline documentation
- [ ] Leave TODO comments for integration
```

### Step 3: Integration Phase (CRITICAL)
```markdown
- [ ] Connect component to calling code
- [ ] Remove all TODO comments
- [ ] Verify component is actually used
- [ ] Check no orphaned instantiation
```

### Step 4: Testing Phase
```markdown
- [ ] Write integration test verifying usage
- [ ] Verify test fails if integration removed
- [ ] Document usage in calling code
```

**Integration Test Template**:

```python
# tests/integration/test_infrastructure_integration.py

def test_training_state_actually_used():
    """Verify TrainingState is integrated with training loop"""
    
    # Setup
    loss = AdaptiveWeightLoss(...)
    initial_state_size = len(loss.state.weight_history)
    
    # Execute training iteration
    loss.objective(preds, train_data)
    
    # CRITICAL: Verify state was updated
    assert len(loss.state.weight_history) > initial_state_size, \
        "TrainingState.weight_history not updated - integration missing!"
    
    # Verify iteration counter incremented
    assert loss.state.iteration > 0, \
        "TrainingState.iteration not incremented - integration missing!"

def test_infrastructure_accumulates_data():
    """Verify infrastructure accumulates across multiple iterations"""
    
    loss = AdaptiveWeightLoss(...)
    
    # Execute multiple iterations
    for i in range(10):
        loss.objective(preds, train_data)
    
    # Verify history length matches iterations
    assert len(loss.state.weight_history) == 10, \
        f"Expected 10 snapshots, got {len(loss.state.weight_history)}"
    
    # Verify data is different (not all same reference)
    unique_count = len(set(map(id, loss.state.weight_history)))
    assert unique_count > 1, \
        "All snapshots are same object - missing copy!"
```

**Code Review Checklist** (For reviewers):

```markdown
## Infrastructure Integration Review

Instantiation Check:
- [ ] Is the component instantiated?
- [ ] Where is it instantiated?
- [ ] Is instantiation necessary?

Usage Check:
- [ ] Are component methods called?
- [ ] Where are they called?
- [ ] What is the call frequency?

Integration Test Check:
- [ ] Does an integration test verify usage?
- [ ] Does test fail if integration removed?
- [ ] Is test meaningful (not just instantiation)?

Documentation Check:
- [ ] Is usage documented in calling code?
- [ ] Are integration points clear?
- [ ] Are TODOs resolved?

Red Flags:
- ⚠️ Component instantiated but methods never called
- ⚠️ Only unit tests exist (no integration tests)
- ⚠️ TODO comments about integration not resolved
- ⚠️ Documentation doesn't mention where it's used
```

**Definition of Done**:

A component is "done" when:
```markdown
✅ Code exists AND
✅ Code is integrated into system AND
✅ Integration test verifies usage AND
✅ Documentation mentions where it's used AND
✅ No TODOs remain about integration

NOT done if:
❌ Code exists but isn't called
❌ Only unit tests (no integration tests)
❌ TODOs about integration remain
❌ Documentation doesn't mention usage
```

---

## Review Process

### Two-Phase Review (MANDATORY)

Refactorings require **TWO separate reviews**:

#### Phase 1: Architecture Review

**Focus**: Code structure, design patterns, organization

**Checklist**:
```markdown
## Architecture Review Checklist

- [ ] Design patterns correctly implemented?
- [ ] Code duplication eliminated?
- [ ] Naming clear and consistent?
- [ ] Documentation adequate?
- [ ] Code follows team standards?
- [ ] Abstractions appropriate?
- [ ] Module boundaries clear?
```

**Approval Criteria**: Architecture review can approve even if algorithm review pending.

---

#### Phase 2: Algorithm Review

**Focus**: Numerical correctness, functional equivalence

**Checklist**:
```markdown
## Algorithm Review Checklist (MANDATORY - BLOCKING)

### Numerical Equivalence
- [ ] All numerical regression tests pass?
- [ ] Tolerance appropriate (typically 1e-6)?
- [ ] Tests run with baseline data?
- [ ] Edge cases tested?

### Learning Verification
- [ ] Model actually learns (metrics improve)?
- [ ] No frozen learning (metrics stuck)?
- [ ] No NaN/Inf in outputs?
- [ ] Weights/parameters evolve correctly?

### Production Characteristics
- [ ] Tested with production data distribution?
- [ ] Handles extreme imbalance?
- [ ] Handles hierarchical constraints?
- [ ] Handles zero variance scenarios?

### Dependency Analysis
- [ ] All dependencies documented?
- [ ] Library compatibility verified?
- [ ] Data format compatibility confirmed?
- [ ] No custom forks overlooked?

### Integration Verification
- [ ] New infrastructure actually used?
- [ ] Integration tests pass?
- [ ] No orphaned code?
- [ ] All TODOs resolved?

### Change Analysis
- [ ] No algorithmic changes during refactoring?
- [ ] No optimizations during Phase 1?
- [ ] Each layer verified independently?
- [ ] Clear commit history?

### Performance
- [ ] Performance within 10% of baseline?
- [ ] If optimized: profile data provided?
- [ ] If optimized: equivalence maintained?
```

**Approval Criteria**: 
- **CANNOT approve** unless ALL items checked
- Both architecture AND algorithm reviews must pass
- Algorithm review is **BLOCKING** - highest priority

---

## Post-Deployment Phase

### Staged Rollout (MANDATORY)

**Objective**: Verify refactored code in production with real traffic before full deployment.

**Rollout Phases**:

#### Phase 1: Shadow Mode (Week 1)
```python
# Both legacy and refactored run, legacy serves traffic

if SHADOW_MODE:
    # Run both implementations
    legacy_output = legacy_model.predict(X)
    refactored_output = refactored_model.predict(X)
    
    # Log any divergence
    if not np.allclose(legacy_output, refactored_output, rtol=1e-4):
        log_divergence(legacy_output, refactored_output)
        alert_team("Numerical divergence detected")
    
    # Use legacy output for production
    return legacy_output
```

**Monitoring**:
- [ ] Numerical divergence rate < 0.1%
- [ ] No NaN/Inf in refactored outputs
- [ ] Latency within 10% of legacy
- [ ] Memory usage comparable

#### Phase 2: Canary Mode (Week 2)
```python
# 1% traffic to refactored, 99% to legacy

if CANARY_MODE:
    if random.random() < 0.01:  # 1% traffic
        return refactored_model.predict(X)
    else:
        return legacy_model.predict(X)
```

**Monitoring**:
- [ ] Error rate identical to legacy
- [ ] Latency P50, P99, P99.9 comparable
- [ ] No customer complaints
- [ ] Metrics match legacy

#### Phase 3: Ramp (Weeks 3-4)
```
Week 3: 10% → 50% refactored traffic
Week 4: 50% → 100% refactored traffic
```

**Monitoring**:
- [ ] A/B test metrics equivalent
- [ ] No increase in errors or latency
- [ ] Customer experience unchanged

#### Phase 4: Full Deployment (Week 5)
```
100% traffic to refactored code
Legacy code deprecated (kept as fallback)
```

#### Phase 5: Cleanup (Week 6+)
```
Remove legacy code after confidence established
Document lessons learned
Archive deployment data
```

**Rollback Plan** (MANDATORY):

```markdown
## Rollback Procedure

### Immediate Rollback Triggers
- Error rate >2x baseline
- Latency P99 >2x baseline
- NaN/Inf in outputs
- Customer complaints

### Rollback Steps
1. Switch traffic to legacy (< 5 minutes)
2. Alert team
3. Preserve logs and metrics
4. Root cause analysis
5. Fix and re-deploy

### Post-Rollback
- [ ] Document what went wrong
- [ ] Identify missed test cases
- [ ] Update test suite
- [ ] Re-verify before re-deploy
```

---

## Common Anti-Patterns to Avoid

### Anti-Pattern 1: "Clever Optimization" During Refactoring

❌ **Problem**: Adding "optimizations" that alter algorithms during refactoring.

```python
# ❌ WRONG: Adding caching during refactoring
def objective(self, preds, train_data):
    if id(preds) in self._cache:  # ← "Optimization"
        return self._cache[id(preds)]
    
    result = compute_gradients(preds)
    self._cache[id(preds)] = result  # ← Breaks correctness
    return result
```

✅ **Solution**: Separate refactoring from optimization.

```python
# ✅ CORRECT: Phase 1 - No optimization
def objective(self, preds, train_data):
    # Always compute fresh (no caching)
    result = compute_gradients(preds)
    return result

# Phase 2: After equivalence verified, profile first
# Only optimize if proven bottleneck (0.3% of time → not worth it)
```

**Prevention**: 
- Mandate "No Optimization During Refactoring" policy
- Pre-commit hooks to detect optimization patterns
- Code review checklist blocks optimizations

---

### Anti-Pattern 2: Assuming "It Compiles" Means "It's Correct"

❌ **Problem**: Relying on compilation and test coverage without numerical verification.

```python
# ❌ Tests that give false confidence
def test_model_trains():
    model.train(X_train, y_train)  # ✓ Completes
    assert True  # ✓ Passes
    
# But model learned nothing! (AUC frozen)
```

✅ **Solution**: Test numerical outputs and learning.

```python
# ✅ CORRECT: Verify actual behavior
def test_model_learns():
    metrics = []
    for i in range(100):
        model.train_iteration(X_train, y_train)
        metric = model.evaluate(X_val, y_val)
        metrics.append(metric)
    
    # Verify learning occurs
    assert metrics[-1] > metrics[0], "Model didn't learn!"
    
    # Verify not frozen
    recent_10 = metrics[-10:]
    assert not all(m == recent_10[0] for m in recent_10), \
        "Metrics frozen!"
```

**Prevention**:
- Mandate numerical regression tests
- Require learning verification tests
- Gate deployments on equivalence tests passing

---

### Anti-Pattern 3: Synthetic Test Data Hides Production Bugs

❌ **Problem**: Testing with unrealistic data that doesn't reveal issues.

```python
# ❌ WRONG: Synthetic balanced data
X_test = np.random.randn(1000, 50)
y_test = np.random.randint(0, 2, size=(1000,))
# Hides: imbalance, hierarchical constraints, edge cases
```

✅ **Solution**: Test with production characteristics.

```python
# ✅ CORRECT: Production-like data
prod_sample = load_production_sample()

# Test extreme imbalance (110 vs 133,969 samples)
test_imbalanced(prod_sample)

# Test hierarchical constraints (if subtask=1 then main=1)
test_hierarchical(prod_sample)

# Test zero variance (constant predictions)
test_edge_cases(prod_sample)
```

**Prevention**:
- Require production-scale test dataset
- Document data characteristics in tests
- Include edge case test suite

---

### Anti-Pattern 4: Skipping Dependency Analysis

❌ **Problem**: Assuming libraries work the same way without verification.

```python
# ❌ WRONG: Assume standard library works
import lightgbm as lgb  # Standard library
dataset = lgb.Dataset(X, label=y)  # y.shape = (n, 6)
# Fails: "label must be 1-dimensional"
```

✅ **Solution**: Verify compatibility before coding.

```markdown
## Pre-Coding Verification (MANDATORY)

1. Document legacy data shapes
   - Labels: (n_samples, n_tasks) = (133969, 6)
   
2. Read library documentation
   - Standard LightGBM: label must be 1D
   
3. Test compatibility
   - Test fails: "label must be 1-dimensional"
   
4. Decision
   - Must use custom fork OR redesign
```

**Prevention**:
- Mandatory pre-coding verification checklist
- Data format documentation template
- Compatibility test before any coding

---

### Anti-Pattern 5: Creating "Zombie" Infrastructure

❌ **Problem**: Creating code that's never integrated or used.

```python
# ❌ WRONG: Created but never used
class TrainingState:
    def __init__(self):
        self.weight_history = []  # ← Created
    
    def update(self, weights):
        self.weight_history.append(weights)  # ← Method exists

# In training loop
loss = AdaptiveWeightLoss()
loss.state = TrainingState()  # ← Instantiated
# ❌ MISSING: loss.state.update(weights)
```

✅ **Solution**: Verify integration with tests.

```python
# ✅ CORRECT: Integration test catches this
def test_state_actually_used():
    loss = AdaptiveWeightLoss()
    loss.objective(preds, train_data)
    
    # Would FAIL if not integrated
    assert len(loss.state.weight_history) > 0  # ✗ FAILS!
```

**Prevention**:
- Mandatory integration verification checklist
- Integration tests required for new infrastructure
- CI/CD checks for unused code

---

## Quick Reference Checklists

### Pre-Refactoring Checklist

```markdown
- [ ] Baseline outputs captured and saved
- [ ] Data format documentation complete
- [ ] All dependencies documented
- [ ] Library compatibility verified
- [ ] Edge cases collected
- [ ] Performance baseline measured
- [ ] Numerical regression tests written
- [ ] All tests pass with legacy code
```

**Gate**: Cannot start refactoring until complete.

---

### Refactoring Checklist (Per Layer)

```markdown
- [ ] One conceptual change only
- [ ] No algorithmic modifications
- [ ] No optimizations
- [ ] All tests pass
- [ ] No NaN/Inf in outputs
- [ ] Learning verified
- [ ] Performance within 10% of baseline
- [ ] Clear commit message
```

**Gate**: Cannot proceed to next layer until complete.

---

### Optimization Checklist (If Needed)

```markdown
- [ ] Phase 1 complete (all tests pass)
- [ ] Profile data shows bottleneck
- [ ] Optimization targets proven bottleneck
- [ ] Expected improvement >20%
- [ ] Equivalence tests pass after optimization
- [ ] Performance actually improved
- [ ] Trade-offs documented
```

**Gate**: Each optimization verified independently.

---

### Deployment Checklist

```markdown
- [ ] Shadow mode: 1 week, <0.1% divergence
- [ ] Canary mode: 1% traffic, no issues
- [ ] Ramp: 10% → 50% → 100%
- [ ] Monitoring alerts configured
- [ ] Rollback procedure documented
- [ ] On-call team informed
```

**Gate**: Cannot proceed without passing each phase.

---

## Summary

This SOP provides a systematic framework for algorithm-preserving refactoring with three key phases:

### Phase 1: Pre-Refactoring (Mandatory)
1. **Establish Baseline**: Capture outputs, document shapes, identify dependencies
2. **Verify Compatibility**: Test libraries, document decisions
3. **Create Tests**: Write numerical regression tests that will verify equivalence

**Key Deliverable**: Comprehensive test suite passing with legacy code

### Phase 2: Refactoring (Mandatory)
1. **Refactor for Correctness**: Zero optimizations, preserve exact behavior
2. **Layer-by-Layer**: One conceptual change per commit
3. **Verify Each Layer**: All tests must pass before proceeding

**Key Deliverable**: Refactored code with verified numerical equivalence

### Phase 3: Optimization (Optional)
1. **Profile First**: Find actual bottlenecks
2. **Optimize Sparingly**: Only proven bottlenecks
3. **Verify Each Change**: Maintain equivalence

**Key Deliverable**: Faster code with maintained correctness (if needed)

### Phase 4: Deployment (Mandatory)
1. **Shadow Mode**: Both run, legacy serves traffic
2. **Canary Mode**: 1% traffic to refactored
3. **Ramp**: Gradual increase to 100%
4. **Monitor**: Alert on divergence, ready to rollback

**Key Deliverable**: Safe production deployment with fallback

---

## Core Principles Summary

1. **Correctness First**: Numerical equivalence is ONLY non-negotiable requirement
2. **Separate Concerns**: Refactoring ≠ Optimization (distinct phases)
3. **Test What Matters**: Numerical outputs, not just code structure
4. **Understand Dependencies**: Verify compatibility BEFORE coding
5. **Refactor in Layers**: One change at a time, verify each
6. **Production Data**: Test with realistic characteristics
7. **Optimize Later**: Profile first, most code doesn't need optimization

---

## Success Criteria

A refactoring is successful when:

✅ **Technical Success**:
- All numerical equivalence tests pass (rtol ≤ 1e-6)
- Model learns (metrics improve over iterations)
- No NaN/Inf values in outputs
- Performance within 10% of baseline
- Edge cases handled correctly

✅ **Process Success**:
- Clear layer-by-layer commit history
- Each layer verified independently
- No algorithmic changes during Phase 1
- Optimizations (if any) verified separately

✅ **Production Success**:
- Shadow mode shows <0.1% divergence
- Canary and ramp phases complete
- No customer impact
- Metrics equivalent to legacy

---

## Related Resources

### Internal Documentation
- **[COE Documentation Guide](./coe_documentation_guide.md)** - Writing post-incident analyses
- **[MTGBM Refactoring COE](../4_analysis/2025-12-19_mtgbm_refactoring_coe.md)** - Real incident example

### Testing Resources
- **[Pytest Best Practices](./pytest_best_practices_and_troubleshooting_guide.md)** - Testing guidelines
- NumPy testing: `numpy.testing.assert_allclose()` documentation
- pytest fixtures for baseline data

### Code Quality
- **[Design Principles](./design_principles.md)** - General design guidance
- **[Standardization Rules](./standardization_rules.md)** - Coding standards

### Examples
- MTGBM refactoring: Good final architecture, cautionary tale of process
- See `projects/cap_mtgbm/dockers/models/` for correct implementation

---

## FAQ

**Q: How long should refactoring take?**
A: Highly variable, but expect:
- Pre-refactoring phase: 2-3 days
- Refactoring phase: 1-2 weeks
- Optimization phase: 1 week (if needed)
- Deployment phase: 1-2 weeks
- Total: 3-6 weeks for complex algorithms

**Q: Can I skip the staged rollout for "small" changes?**
A: No. Staged rollout is mandatory for all algorithm refactoring. Even "small" changes can have large impacts (see MTGBM COE).

**Q: What if standard library doesn't support my data format?**
A: Three options:
1. Use custom fork (if maintained and tested)
2. Redesign architecture to match library requirements
3. Contribute upstream to library
Decision depends on effort/risk trade-off.

**Q: When should I optimize?**
A: Only after:
- Phase 1 complete (all tests pass)
- Profile shows actual bottleneck
- Improvement >20%
- You've verified simpler alternatives

Most refactorings need ZERO optimization.

**Q: What if I find a bug during refactoring?**
A: Two options:
1. Fix in legacy first, then refactor
2. Document as "bug fix" (not refactoring), verify intentional change

Never silently change behavior during refactoring.

**Q: How do I handle performance regression?**
A: If refactored code is >10% slower:
1. Profile to find cause
2. Optimize proven bottlenecks only
3. Verify equivalence maintained
4. If still slow, consider architecture change

Don't sacrifice correctness for speed.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-19 | Initial SOP created from MTGBM refactoring lessons | ML Platform Team |

---

## Feedback and Improvements

This SOP is a living document. To suggest improvements:

1. Create an issue documenting:
   - What worked well
   - What didn't work
   - Suggested changes
   
2. Update based on learnings from each refactoring

3. Review quarterly with team

**Maintainer**: ML Platform Team  
**Last Review**: 2025-12-19  
**Next Review**: 2026-03-19

---

*This SOP is derived from lessons learned in the MTGBM refactoring incident (Dec 2025). It codifies best practices for maintaining algorithmic correctness while improving code quality.*
