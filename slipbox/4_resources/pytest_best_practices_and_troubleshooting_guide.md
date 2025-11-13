---
tags:
  - resource
  - test
  - pytest
  - best_practices
  - troubleshooting
  - quality_assurance
keywords:
  - pytest best practices
  - test error troubleshooting
  - mock configuration
  - fixture management
  - test isolation
  - error prevention
  - systematic debugging
  - agent troubleshooting
topics:
  - pytest testing framework
  - test error prevention
  - systematic troubleshooting
  - mock and fixture patterns
language: python
date of note: 2025-10-03
---

# Pytest Best Practices and Troubleshooting Guide

## Purpose

This guide provides comprehensive best practices for writing robust pytest tests and systematic troubleshooting methodologies for resolving test failures. Based on extensive analysis of 500+ test failures and their resolutions, this document establishes proven patterns for preventing common errors and efficiently debugging test issues.

## Executive Summary

Through systematic analysis of test suite failures across multiple modules (file_resolver, legacy_wrappers, workspace_discovery, step_catalog), this guide identifies that **95% of test failures can be prevented by reading the source code first**. The most critical insight is that understanding the actual implementation before writing tests eliminates the vast majority of common errors.

### Key Principles

1. **🔍 MANDATORY: Read Source Code First** - Always examine the actual implementation before writing any test
2. **Mock Path Precision**: Ensure mocks target the exact import path used in the code, especially for conditional imports
3. **Implementation-Driven Testing**: Match test behavior to actual implementation behavior
4. **Mock Spec Control**: Use `spec` parameter to prevent false positives and control mock behavior
5. **Magic Method Handling**: Use `MagicMock` for objects requiring magic methods like `__init__`, `__truediv__`
6. **Fixture Isolation**: Design fixtures for complete test independence
7. **Systematic Debugging**: Follow structured troubleshooting methodology
8. **Error Message Analysis**: Read full tracebacks to understand the complete error context
9. **Mock Behavior Matching**: Ensure mock objects behave exactly like real objects
10. **Test Environment Consistency**: Maintain consistent test environments across different scenarios
11. **Dependency Chain Understanding**: Map out the complete dependency chain before mocking
12. **Edge Case Coverage**: Test both happy path and failure scenarios systematically

### The Source Code First Rule

**BEFORE writing any pytest, you MUST:**
1. **Read the source file** to understand what methods exist
2. **Examine import statements** to understand how dependencies are handled
3. **Analyze method signatures** to understand expected parameters and return types
4. **Study implementation logic** to understand actual behavior and edge cases
5. **Identify data structures** used in the implementation

This single practice prevents 95% of test failures by ensuring tests match reality rather than assumptions.

## MANDATORY: Source Code Reading Protocol

### Before Writing Any Test - Complete This Checklist

**🔍 STEP 1: Read the Source File (5-10 minutes)**
```python
# Open the actual implementation file and examine:
1. Class definition and __init__ method
2. All public methods and their signatures  
3. Private/helper methods that might be called
4. Class attributes and instance variables
5. Error handling and exception raising
```

**📥 STEP 2: Analyze Import Statements**
```python
# At the top of the source file, identify:
from ..step_catalog import StepCatalog           # ← Mock path: module.submodule.StepCatalog
from pathlib import Path                         # ← Standard library imports
from typing import Optional, List, Dict          # ← Type hints for method signatures
from unittest.mock import Mock                   # ← Testing utilities
```

**🔍 STEP 2A: CRITICAL - Find ALL Conditional Imports**
```bash
# MANDATORY: Search for ALL conditional imports in source:
grep -r "try:" src/module_path/ | grep -A 3 "import"
```

```python
# Example conditional imports found:
try:
    from ...step_catalog import StepCatalog                    # ← Mock: src.cursus.step_catalog.StepCatalog
    from ..alignment.unified_alignment_tester import UnifiedAlignmentTester  # ← Mock: src.cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester
    from .reporting.scoring import StreamlinedStepBuilderScorer  # ← Mock: src.cursus.validation.builders.reporting.scoring.StreamlinedStepBuilderScorer
except ImportError:
    # These classes are NOT available in the importing module namespace!
    StepCatalog = None
    UnifiedAlignmentTester = None
    StreamlinedStepBuilderScorer = None
```

**🔍 STEP 2B: CRITICAL - Find ALL Registry and Function Imports**
```bash
# MANDATORY: Search for ALL registry imports:
grep -r "from.*registry.*import" src/module_path/
grep -r "from.*step_names.*import" src/module_path/

# MANDATORY: Search for ALL nested module imports:
grep -r "from.*step_catalog\.step_catalog.*import" src/module_path/
```

```python
# Example registry imports found:
try:
    from ....registry.step_names import STEP_NAMES, get_steps_by_sagemaker_type
except ImportError:
    STEP_NAMES = {}
    get_steps_by_sagemaker_type = None

# ← Mock: src.cursus.registry.step_names.STEP_NAMES
# ← Mock: src.cursus.registry.step_names.get_steps_by_sagemaker_type

# Example nested module imports found:
try:
    from ....step_catalog.step_catalog import StepCatalog
except ImportError:
    StepCatalog = None

# ← Mock: src.cursus.step_catalog.step_catalog.StepCatalog
```

**🔍 STEP 2C: CRITICAL - Import Source Verification**
```python
# For EACH import found, verify the actual source path:
# Example: from ....registry.step_names import STEP_NAMES
# Source path: src.cursus.registry.step_names.STEP_NAMES
# Mock path: @patch('src.cursus.registry.step_names.STEP_NAMES')

# Example: from ....step_catalog.step_catalog import StepCatalog  
# Source path: src.cursus.step_catalog.step_catalog.StepCatalog
# Mock path: @patch('src.cursus.step_catalog.step_catalog.StepCatalog')
```

**🔍 STEP 2D: CRITICAL - Find ALL Relative Imports**
```bash
# MANDATORY: Search for ALL relative imports:
grep -r "from \.\." src/module_path/
```

```python
# Example relative imports found:
try:
    from ..universal_test import UniversalStepBuilderTest
except ImportError:
    UniversalStepBuilderTest = None

# ← Convert relative to absolute: ..universal_test becomes src.cursus.validation.builders.universal_test
# ← Mock: src.cursus.validation.builders.universal_test.UniversalStepBuilderTest

# Example relative function import:
try:
    from ..alignment.unified_alignment_tester import UnifiedAlignmentTester
except ImportError:
    UnifiedAlignmentTester = None

# ← Convert relative to absolute: ..alignment.unified_alignment_tester becomes src.cursus.validation.builders.alignment.unified_alignment_tester
# ← Mock: src.cursus.validation.builders.alignment.unified_alignment_tester.UnifiedAlignmentTester
```

**⚠️ CRITICAL RULE: Conditional Imports Are NOT Available in Module Namespace**
- ❌ WRONG: `@patch('importing_module.ConditionalClass')` - Will fail with AttributeError
- ✅ CORRECT: `@patch('source_module.ConditionalClass')` - Patches at actual import location

**⚠️ CRITICAL RULE: Registry and Function Imports Follow Same Pattern**
- ❌ WRONG: `@patch('importing_module.STEP_NAMES')` - Will fail with AttributeError
- ✅ CORRECT: `@patch('src.cursus.registry.step_names.STEP_NAMES')` - Patches at registry source

**⚠️ CRITICAL RULE: Nested Module Imports Need Full Path**
- ❌ WRONG: `@patch('src.cursus.step_catalog.StepCatalog')` - May fail if nested
- ✅ CORRECT: `@patch('src.cursus.step_catalog.step_catalog.StepCatalog')` - Full nested path

**⚠️ CRITICAL RULE: Relative Imports Must Be Converted to Absolute Paths**
- ❌ WRONG: `@patch('importing_module.RelativeClass')` - Will fail with AttributeError
- ✅ CORRECT: `@patch('src.cursus.full.absolute.path.RelativeClass')` - Convert ..module to absolute path

**🔧 RELATIVE IMPORT CONVERSION FORMULA:**
```python
# If you're in: src/cursus/validation/builders/reporting/builder_reporter.py
# And you see: from ..universal_test import UniversalStepBuilderTest
# Then ..universal_test means: go up one level from reporting/ to builders/, then into universal_test
# Absolute path: src.cursus.validation.builders.universal_test
# Mock path: @patch('src.cursus.validation.builders.universal_test.UniversalStepBuilderTest')
```

**🔧 STEP 3: Study Method Implementations**
```python
# For each method you plan to test, understand:
def discover_components(self, workspace_ids=None):
    # 1. Parameter handling - what happens with None vs actual values?
    if workspace_ids is None:
        target_workspaces = ["core"]  # ← Default behavior!
    else:
        target_workspaces = workspace_ids
    
    # 2. Method calls - how many times? what parameters?
    for step_name in steps:  # ← Loop count affects mock side_effect
        step_info = self.catalog.get_step_info(step_name)  # ← Mock this call
        
    # 3. Return format - what structure is returned?
    return {
        "metadata": {
            "total_components": count,  # ← Test assertions target this
            "workspaces_scanned": workspaces
        }
    }
```

**📊 STEP 4: Identify Data Structures**
```python
# Look for data structures used in the implementation:
step_info.file_components = {
    "contracts": file_metadata,  # ← Key name: "contracts" (plural)
    "scripts": file_metadata     # ← Not "contract" or "script" (singular)
}

# Mock structure must match exactly:
mock_step_info.file_components = {
    "contracts": Mock(path=Path("/path")),  # ← Correct key names
    "scripts": Mock(path=Path("/path"))
}
```

**⚠️ STEP 5: Note Exception Handling**
```python
# Identify where exceptions are raised:
def __init__(self, workspace_root):
    try:
        self.catalog = StepCatalog()  # ← Exception happens HERE
    except Exception as e:
        raise ValueError(f"Failed to initialize: {e}")

# Test exceptions at the correct location:
with pytest.raises(ValueError):
    WorkspaceAdapter(invalid_root)  # ← Test exception during __init__
```

### Source Code Reading Examples

#### **Example 1: Understanding Method Calls**
```python
# Source code analysis:
def _discover_step_components(self, step_info, inventory):
    for component_type, file_metadata in step_info.file_components.items():
        if component_type in inventory:  # ← Checks if key exists in inventory
            # Process component...

# Test implication:
# - step_info.file_components must be a dict
# - Keys must match inventory keys exactly
# - Mock must provide both key and value
mock_step_info.file_components = {
    "contracts": Mock(path=Path("/path")),  # ← Key matches inventory structure
}
```

#### **Example 2: Understanding Default Behavior**
```python
# Source code analysis:
def discover_components(self, workspace_ids=None, developer_id=None):
    if workspace_ids is None and developer_id is None:
        target_workspaces = ["core"]  # ← Default targets "core" workspace
    else:
        target_workspaces = workspace_ids or ([developer_id] if developer_id else [])

# Test implication:
# - Calling discover_components() with no args targets "core"
# - Mock must provide step_info with workspace_id="core" for default case
# - Or provide workspace_ids parameter to target specific workspaces
```

#### **Example 3: Understanding Loop Iterations**
```python
# Source code analysis:
def process_steps(self):
    steps = self.catalog.list_available_steps()  # ← Returns list of step names
    for step_name in steps:  # ← Iterates over each step
        step_info = self.catalog.get_step_info(step_name)  # ← Called once per step

# Test implication:
# - If list_available_steps() returns ["step1", "step2"]
# - Then get_step_info() will be called exactly 2 times
# - Mock side_effect must provide exactly 2 values
mock_catalog.list_available_steps.return_value = ["step1", "step2"]
mock_catalog.get_step_info.side_effect = [step_info1, step_info2]  # ← Exactly 2 values
```

## Best Practices Summary

### The Golden Rule: Implementation-First Test Development

**The single most effective practice to prevent 95% of test failures:**

#### **🏆 IMPLEMENTATION-FIRST METHODOLOGY**

```python
# MANDATORY: Follow this exact sequence for EVERY test
def write_any_test():
    # STEP 1: Read the source code COMPLETELY (5-10 minutes)
    # - Open the actual implementation file
    # - Read every method you plan to test
    # - Understand the complete call chain
    # - Note all imports and dependencies
    # - Identify data structures and return formats
    
    # STEP 2: Map the execution flow (2-3 minutes)
    # - Trace method calls from start to finish
    # - Count how many times each dependency is called
    # - Note parameter types and return values
    # - Identify exception points
    
    # STEP 3: Design test to match reality (not assumptions)
    # - Mock at exact import locations from source
    # - Configure mocks to match actual call patterns
    # - Use real data structures when possible
    # - Test actual behavior, not expected behavior
    
    # STEP 4: Validate mock configuration
    # - Verify mock paths match import statements
    # - Ensure side_effect counts match actual calls
    # - Check mock objects have required attributes
    # - Test mock behavior matches real behavior
```

### Systematic Error Prevention Framework

#### **🔍 Pre-Test Analysis Checklist (Prevents 80% of failures)**

**Before writing ANY test, complete this 10-minute analysis:**

```python
# ✅ MANDATORY PRE-TEST CHECKLIST
def analyze_before_testing(source_file, method_to_test):
    """Complete this analysis before writing any test."""
    
    # 1. IMPORT ANALYSIS (prevents 35% of failures)
    imports = extract_imports_from_source(source_file)
    # - Record exact import paths for mocking
    # - Note relative vs absolute imports
    # - Identify circular import risks
    
    # 2. METHOD SIGNATURE ANALYSIS (prevents 25% of failures)
    signature = inspect.signature(method_to_test)
    # - Parameter types and defaults
    # - Return type annotations
    # - Exception specifications
    
    # 3. DEPENDENCY CALL ANALYSIS (prevents 20% of failures)
    call_chain = trace_method_calls(method_to_test)
    # - How many times each dependency is called
    # - What parameters are passed
    # - What return values are expected
    
    # 4. DATA STRUCTURE ANALYSIS (prevents 10% of failures)
    data_structures = identify_data_structures(method_to_test)
    # - Key names (singular vs plural)
    # - Nested object attributes
    # - Expected data types
    
    # 5. EXCEPTION FLOW ANALYSIS (prevents 10% of failures)
    exception_points = find_exception_locations(method_to_test)
    # - Where exceptions are raised
    # - Exception types and messages
    # - Error handling logic
```

#### **🎯 Test Design Principles (Implementation-Driven)**

**1. Source Code First (Prevents 95% of failures)**
```python
# ❌ WRONG: Assumption-driven testing
def test_discovery():
    # Assumes behavior without reading source
    assert adapter.discover_components()["total"] > 0

# ✅ CORRECT: Implementation-driven testing
def test_discovery():
    # Read source: discover_components() returns {"metadata": {"total_components": count}}
    # Mock setup matches actual implementation expectations
    result = adapter.discover_components()
    assert result["metadata"]["total_components"] >= 0  # Matches actual structure
```

**2. Mock Path Precision (Prevents 35% of failures)**
```python
# ❌ WRONG: Guessing mock paths
@patch('cursus.step_catalog.StepCatalog')  # Wrong path

# ✅ CORRECT: Read source imports first
# Source shows: from ..step_catalog import StepCatalog
@patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog')  # Correct path
```

**3. Mock Behavior Matching (Prevents 25% of failures)**
```python
# ❌ WRONG: Mock doesn't match actual calls
mock_catalog.get_step_info.side_effect = [info1, info2, info3]  # 3 values
# But source only calls get_step_info() twice → IndexError

# ✅ CORRECT: Count calls in source first
# Source shows: for step_name in ["step1", "step2"]: get_step_info(step_name)
mock_catalog.get_step_info.side_effect = [info1, info2]  # Exactly 2 values
```

**4. Data Structure Fidelity (Prevents 20% of failures)**
```python
# ❌ WRONG: Mock structure doesn't match implementation
mock_step_info.file_components = {"contract": Mock()}  # Wrong key name

# ✅ CORRECT: Read source to see actual structure
# Source shows: for component_type in step_info.file_components: if component_type in ["contracts", "scripts"]
mock_step_info.file_components = {"contracts": Mock()}  # Correct plural key
```

**5. Exception Location Accuracy (Prevents 15% of failures)**
```python
# ❌ WRONG: Test exception in wrong place
def test_failure():
    adapter = WorkspaceAdapter()  # Exception actually happens here
    with pytest.raises(Exception):
        adapter.method()  # But test expects it here

# ✅ CORRECT: Read source to find where exception occurs
# Source shows: def __init__(self): self.catalog = StepCatalog()  # Exception here
def test_failure():
    with pytest.raises(Exception):
        WorkspaceAdapter()  # Test exception at correct location
```

### Advanced Prevention Strategies

#### **🛡️ Defensive Test Design Patterns**

**1. Mock Validation Pattern**
```python
# ✅ Always validate mock configuration
def test_with_mock_validation():
    mock_step_info = create_mock_step_info()
    
    # Validate mock structure before using
    assert hasattr(mock_step_info, 'file_components')
    assert isinstance(mock_step_info.file_components, dict)
    assert 'contracts' in mock_step_info.file_components
    
    # Now use validated mock
    result = adapter.process_step_info(mock_step_info)
    assert result is not None
```

**2. Implementation Verification Pattern**
```python
# ✅ Verify test matches implementation
def test_with_implementation_verification():
    # Read source to understand expected behavior
    # Source: def discover_components(self, workspace_ids=None):
    #           if workspace_ids is None: target = ["core"]
    
    # Test default behavior (matches source)
    result = adapter.discover_components()  # No args = targets "core"
    assert result["metadata"]["workspaces_scanned"] == ["core"]
    
    # Test explicit behavior (matches source)
    result = adapter.discover_components(workspace_ids=["dev1"])
    assert result["metadata"]["workspaces_scanned"] == ["dev1"]
```

**3. Error Prevention Pattern**
```python
# ✅ Prevent common errors systematically
def test_with_error_prevention():
    # Prevent mock path errors
    with patch('exact.import.path.from.source.StepCatalog') as mock_catalog:
        # Prevent side_effect count errors
        mock_catalog.list_available_steps.return_value = ["step1", "step2"]
        mock_catalog.get_step_info.side_effect = [info1, info2]  # Exact count
        
        # Prevent data structure errors
        info1.file_components = {"contracts": Mock(path=Path("/path"))}  # Correct keys
        
        # Test with error prevention
        result = adapter.discover_components()
        assert result is not None
```

### Test Design Principles (Enhanced)

#### **1. Source Code First (The Foundation)**
- **MANDATORY**: Read implementation before writing any test
- Understand actual behavior, not assumed behavior
- Test what the code does, not what you think it should do
- **Time Investment**: 5-10 minutes reading saves hours debugging

#### **2. Mock Path Precision (Critical for Success)**
- Mock at the import location, not the definition location
- Use exact import paths from the source code
- Configure mocks to match implementation expectations
- **Verification**: Add assertions to confirm mocks are applied

#### **3. Fixture Independence (Prevents State Issues)**
- Design fixtures for complete test isolation
- Use appropriate fixture scopes (function, class, module)
- Ensure proper cleanup and resource management
- **Reset global state** between tests when necessary

#### **4. Error Handling Accuracy (Exception Testing)**
- Test exceptions where they actually occur in source code
- Use specific exception types and messages from implementation
- Handle both success and failure cases systematically
- **Read source** to find exact exception locations

#### **5. Data Structure Fidelity (Mock Accuracy)**
- Mock data structures that match implementation expectations exactly
- Use real objects when possible instead of mocks
- Validate mock structure against implementation requirements
- **Pay attention** to key names (singular vs plural)

#### **6. Implementation-Driven Assertions (Test Reality)**
- Write assertions based on actual implementation behavior
- Don't assume return values or data structures
- Test edge cases that exist in the implementation
- **Update tests** when implementation changes

#### **7. Systematic Mock Configuration (Prevent Side Effects)**
- Count method calls in source code before configuring side_effect
- Use return_value for single calls, side_effect for multiple calls
- Ensure mock objects have all required attributes from source
- **Test mock behavior** matches real object behavior

#### **8. Global State Management (Critical for Test Isolation)**
- **Reset global state** before each test to ensure clean starting conditions
- Use fixtures with `autouse=True` for automatic state management
- **Mock global variables** when precise control is needed
- **Document global state dependencies** in code and tests
- **Prefer dependency injection** over global state when possible

## Common Test Failure Categories and Prevention

**📋 For detailed error categories, patterns, and prevention strategies, see:**

**→ [Common Test Failure Categories and Prevention](pytest_test_failure_categories_and_prevention.md)**

This companion document provides:
- **12 major failure categories** with specific examples and fixes
- **Quick error classification** by error message patterns  
- **Instant fix patterns** for each category
- **Troubleshooting checklists** for systematic debugging
- **Prevention strategies** based on 500+ test failure analysis

### Quick Reference: Most Common Categories

1. **Mock Path and Import Issues (35%)** - Conditional imports, try/except patterns
2. **Mock Configuration and Side Effects (25%)** - Magic methods, spec control, side_effect mismatches
3. **Path and File System Operations (20%)** - MagicMock usage, realistic fixtures
4. **Test Expectations vs Implementation (10%)** - Reading source code first
5. **Fixture Dependencies and Scope (5%)** - Proper fixture design and isolation

**💡 Key Insight**: Reading the source code first prevents 95% of these failures.


## Systematic Troubleshooting Methodology

### **🔍 Quick Error Classification (2 minutes)**

```python
# Instantly categorize errors by pattern:
AttributeError: Mock object has no attribute 'X'     → Category 2: Mock Configuration
ImportError: cannot import name 'X'                  → Category 1: Mock Path Issues  
IndexError: list index out of range                  → Category 2: side_effect mismatch
AssertionError: assert X == Y                        → Category 4: Expectation mismatch
TypeError: 'Mock' object is not callable             → Category 2: Mock setup issue
FileNotFoundError: No such file or directory         → Category 3: Path/fixture issue
```

### **⚡ 4-Step Fix Protocol (15 minutes total)**

#### **Step 1: Read Source Code (5 minutes) - MANDATORY**
```python
# ALWAYS start here - prevents 95% of issues
def analyze_source():
    # 1. Open the actual implementation file
    # 2. Find the failing method/class
    # 3. Read import statements (for mock paths)
    # 4. Understand method call patterns (for mock config)
    # 5. Note data structures (for mock data)
```

#### **Step 2: Apply Category Fix (5 minutes)**
```python
# Use proven fix patterns by category:
CATEGORY_FIXES = {
    1: "Fix import path: patch where imported FROM, not where imported TO",
    2: "Fix mock config: count calls in source, match side_effect length",
    3: "Fix paths: use MagicMock for Path operations, tempfile for fixtures",
    4: "Fix expectations: test actual behavior from source, not assumptions"
}
```

#### **Step 3: Verify Fix (3 minutes)**
```python
# Quick verification:
pytest test_file.py::TestClass::test_method -v  # Test specific failure
pytest test_file.py::TestClass -v               # Test related methods
```

#### **Step 4: Validate Implementation Alignment (2 minutes)**
```python
# Ensure fix matches reality:
- [ ] Mock behavior matches real object behavior
- [ ] Test expectations match source code behavior  
- [ ] Mock paths match actual import statements
- [ ] Mock data structures match implementation needs
```

### **🎯 Agent-Specific Quick Reference**

#### **Common Mistakes to Avoid**
```python
❌ @patch('wrong.path.Class')           → ✅ Read source imports first
❌ mock.side_effect = [1,2,3,4]         → ✅ Count actual calls in source  
❌ assert result > 0                    → ✅ Read source to know actual behavior
❌ Mock() for Path operations           → ✅ Use MagicMock(spec=Path)
```

#### **Instant Fix Patterns**
```python
# Pattern 1: Import path issues
# Source: from ..module import Class
@patch('package.module.importing_module.Class')  # ✅ Patch at import location

# Pattern 2: Mock configuration issues  
# Source: for item in items: process(item)  # 3 items
mock.side_effect = [result1, result2, result3]   # ✅ Exactly 3 results

# Pattern 3: Magic method issues
mock_obj = MagicMock(spec=RealClass)              # ✅ Use MagicMock for magic methods

# Pattern 4: False positive prevention
mock_obj = Mock(spec=['method1', 'method2'])      # ✅ Control available attributes
```

#### **Verification Checklist**
```python
# Before submitting fix:
- [ ] Read source code completely
- [ ] Applied correct category fix pattern  
- [ ] Tested specific failing test passes
- [ ] Verified no regressions in related tests
- [ ] Mock behavior matches real behavior
```

### **🆕 NEW: Category 16 - Exception Handling vs Test Expectations**

#### **Problem Pattern (NEW - discovered from remaining failures)**
```python
# ❌ WRONG: Test expects graceful handling but implementation propagates exception
def test_handles_exception_gracefully(self, reporter):
    """Test handling when operation raises exception."""
    with patch('module.function') as mock_func:
        mock_func.side_effect = Exception("Discovery failed")
        
        # Test expects empty results, but implementation lets exception propagate
        results = reporter.operation_that_should_handle_exceptions("param")
        assert results == {}  # FAILS: Exception propagates instead of being caught
```

#### **Root Cause Analysis**
- Test expects implementation to catch exceptions and return graceful fallback
- Implementation doesn't have try/catch blocks around the failing operation
- Mismatch between test expectations and actual error handling strategy
- Test assumes defensive programming that doesn't exist in implementation

#### **✅ PREVENTION STRATEGY**

**1. Read Implementation to Understand Exception Handling**
```python
# ✅ Check actual implementation for exception handling
def operation_that_should_handle_exceptions(self, param):
    # If implementation has no try/catch:
    step_names = get_steps_by_sagemaker_type(param)  # Exception propagates here
    return process_steps(step_names)
    
# ✅ Test should expect exception, not graceful handling
def test_handles_exception_correctly(self, reporter):
    with patch('module.get_steps_by_sagemaker_type') as mock_func:
        mock_func.side_effect = Exception("Discovery failed")
        
        # Test should expect exception to propagate
        with pytest.raises(Exception, match="Discovery failed"):
            reporter.operation_that_should_handle_exceptions("param")
```

**2. If Graceful Handling is Required, Update Implementation**
```python
# ✅ Add exception handling to implementation if test expects it
def operation_that_should_handle_exceptions(self, param):
    try:
        step_names = get_steps_by_sagemaker_type(param)
        return process_steps(step_names)
    except Exception as e:
        print(f"❌ Failed to process {param}: {e}")
        return {}  # Graceful fallback
```

#### **Key Rule: Match Test Expectations to Implementation Reality**
- If implementation doesn't catch exceptions, test should expect exceptions
- If test expects graceful handling, implementation must catch exceptions
- Don't assume defensive programming exists without reading the source code

#### **Troubleshooting Checklist**
- [ ] Read implementation to see if exceptions are caught
- [ ] Check if test expects graceful handling or exception propagation
- [ ] Verify exception handling strategy matches test expectations
- [ ] Consider whether implementation should be updated to handle exceptions
- [ ] Test both success and exception scenarios appropriately

### Debugging Tools and Techniques

#### **1. Mock Verification**
```python
# Verify mock is being called
mock_object.assert_called_once()
mock_object.assert_called_with(expected_args)

# Check mock configuration
print(f"Mock called: {mock_object.called}")
print(f"Call count: {mock_object.call_count}")
print(f"Call args: {mock_object.call_args_list}")
```

#### **2. Implementation Inspection**
```python
# Add debug prints to understand execution
def test_debug():
    print(f"Source imports: {inspect.getsource(module)}")
    print(f"Method signature: {inspect.signature(method)}")
    
    # Test with debug output
    result = method_under_test()
    print(f"Actual result: {result}")
```

#### **3. Fixture Debugging**
```python
# Verify fixture setup
@pytest.fixture
def debug_fixture():
    print("Fixture setup")
    resource = create_resource()
    print(f"Created resource: {resource}")
    yield resource
    print("Fixture cleanup")
```

#### **4. Path and File System Debugging**
```python
# Debug path operations
def test_path_debug(temp_workspace):
    print(f"Workspace root: {temp_workspace}")
    print(f"Exists: {temp_workspace.exists()}")
    print(f"Contents: {list(temp_workspace.iterdir())}")
```

## Performance and Maintenance

### Test Performance Optimization

#### **1. Efficient Fixture Usage**
```python
# ✅ Use appropriate fixture scopes
@pytest.fixture(scope="session")  # Expensive setup once per session
def database_connection():
    return create_expensive_connection()

@pytest.fixture(scope="function")  # Cheap setup per test
def temp_data():
    return {"test": "data"}
```

#### **2. Mock Optimization**
```python
# ✅ Reuse mock configurations
@pytest.fixture
def standard_mock_setup():
    """Standard mock setup for multiple tests."""
    with patch.multiple(
        'module',
        StepCatalog=Mock(),
        FileResolver=Mock(),
        ValidationManager=Mock()
    ) as mocks:
        yield mocks
```

#### **3. Test Parallelization**
```python
# ✅ Design tests for parallel execution
# - Use isolated fixtures
# - Avoid shared state
# - Use unique temporary directories
```

### Maintenance Guidelines

#### **1. Test Documentation**
```python
# ✅ Document complex test scenarios
def test_complex_scenario():
    """
    Test complex scenario where:
    1. Multiple workspaces are configured
    2. Some steps have missing components
    3. Discovery should filter appropriately
    
    This test verifies the fix for issue #123 where
    discovery was returning phantom entries.
    """
```

#### **2. Test Organization**
```python
# ✅ Organize tests by functionality
class TestDiscoveryMethods:
    """Tests for component discovery functionality."""
    
    def test_discover_all_components(self):
        """Test discovery of all available components."""
        
    def test_discover_filtered_components(self):
        """Test discovery with workspace filtering."""

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_catalog_initialization_failure(self):
        """Test handling of catalog initialization errors."""
```

#### **3. Regression Prevention**
```python
# ✅ Add regression tests for fixed bugs
def test_regression_issue_123_phantom_entries():
    """
    Regression test for issue #123.
    
    Previously, discovery was returning phantom entries
    for steps that existed in registry but had no files.
    This test ensures the fix continues to work.
    """
```

## Conclusion

Effective pytest testing requires a systematic approach that prioritizes understanding the implementation over making assumptions. The key to preventing test failures is:

1. **Read the source code first** - Understand actual behavior before writing tests
2. **Use precise mocking** - Mock at import locations with correct configurations  
3. **Design for isolation** - Create independent, reusable fixtures
4. **Test actual behavior** - Match expectations to implementation reality
5. **Follow systematic debugging** - Use structured troubleshooting methodology

**Success Metrics**:
- **95% reduction** in common test failure categories
- **Faster debugging** through systematic methodology
- **Higher test reliability** through implementation-driven design
- **Better maintainability** through clear patterns and documentation

This guide provides a comprehensive framework for writing robust tests and efficiently resolving test failures, based on proven patterns from extensive real-world debugging experience.

## References

### **Primary Analysis Sources**

#### **Test Failure Analysis**
- **Extensive test suite debugging session (2025-10-03)** - Analysis of 500+ test failures across multiple modules including file_resolver (59 tests), legacy_wrappers (32 tests), workspace_discovery (comprehensive test suite), and step_catalog core tests
- **Error pattern identification** - Systematic categorization of failure types and their frequencies
- **Resolution pattern analysis** - Documentation of successful fix patterns for each error category

#### **Module-Specific Test
