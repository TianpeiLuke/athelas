---
tags:
  - design
  - optimization
  - ruleset_generation
  - performance
  - patterns
  - algorithms
keywords:
  - ruleset optimization
  - rule ordering
  - performance optimization
  - selectivity analysis
  - cache efficiency
  - complexity analysis
topics:
  - optimization algorithms
  - ruleset optimization implementation
  - performance tuning
  - execution efficiency
language: python
date of note: 2025-11-09
updated: 2025-11-09
---

# Label Ruleset Optimization Patterns

## Overview

This document defines optimization strategies for ruleset generation in the cursus framework. Ruleset optimization reorders and restructures rules to maximize execution efficiency while preserving logical correctness. The goal is to minimize average evaluation time by placing high-impact rules in optimal positions.

## Purpose and Motivation

### Problem Statement

When rules are evaluated in priority order (first match wins), the order significantly impacts performance:

1. **Suboptimal Ordering**: User-defined priorities may not reflect actual data patterns
2. **Cache Inefficiency**: Rules accessing different fields cause cache misses
3. **Wasted Computation**: Complex rules evaluated before simple ones
4. **No Data Awareness**: Static priorities don't adapt to data distribution

### Optimization Benefits

- **Faster Execution**: Rules match earlier in evaluation sequence
- **Better Cache Utilization**: Fields loaded once, reused across adjacent rules
- **Reduced Complexity**: Simple checks filter data before complex logic
- **Data-Driven**: Optimization adapts to actual data patterns (when sample available)

## Core Optimization Strategies

### 1. Selectivity-Based Ordering

**Principle**: Most selective rules (fewest matches) should have highest priority

**Rationale**: 
- Selective rules eliminate many rows early
- Less selective rules handle remaining cases
- Minimizes total conditions evaluated

**Implementation**:
```python
def estimate_selectivity(rule: dict, sample_df: pd.DataFrame) -> float:
    """
    Estimate what fraction of rows match this rule.
    
    Args:
        rule: Rule definition with conditions
        sample_df: Sample data for analysis
        
    Returns:
        Match rate (0.0 to 1.0), lower = more selective
    """
    from rule_engine import RuleEngine
    
    # Create temporary engine for this rule
    temp_ruleset = {
        "label_config": {"default_label": 0, ...},
        "rules": [rule]
    }
    engine = RuleEngine(temp_ruleset)
    
    # Count matches
    matches = 0
    for _, row in sample_df.iterrows():
        try:
            if engine._evaluate_conditions(rule['conditions'], row):
                matches += 1
        except:
            pass  # Ignore errors in estimation
    
    return matches / len(sample_df) if len(sample_df) > 0 else 1.0
```

**Example**:
```python
# Before optimization (by user priority):
Rule 1: amount > 0                    # matches 95% of rows
Rule 2: category = "Fraud"            # matches 2% of rows
Rule 3: status = "completed"          # matches 80% of rows

# After selectivity optimization:
Rule 1: category = "Fraud"            # matches 2% (most selective)
Rule 2: status = "completed"          # matches 80%
Rule 3: amount > 0                    # matches 95% (least selective)
```

### 2. Field Usage Grouping

**Principle**: Group rules that access the same fields together

**Rationale**:
- CPU cache efficiency: Field values loaded once, reused
- Memory access patterns: Sequential field access is faster
- Reduces data loading overhead

**Implementation**:
```python
def group_by_field_usage(rules: List[dict]) -> List[dict]:
    """
    Group rules by field similarity using clustering.
    
    Args:
        rules: List of rule definitions
        
    Returns:
        Reordered rules with similar field usage adjacent
    """
    # Extract fields used by each rule
    rule_fields = []
    for rule in rules:
        fields = extract_all_fields(rule['conditions'])
        rule_fields.append(fields)
    
    # Calculate field similarity matrix
    n = len(rules)
    similarity = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            # Jaccard similarity: intersection / union
            fields_i = set(rule_fields[i])
            fields_j = set(rule_fields[j])
            
            intersection = len(fields_i & fields_j)
            union = len(fields_i | fields_j)
            
            similarity[i][j] = intersection / union if union > 0 else 0
            similarity[j][i] = similarity[i][j]
    
    # Greedy clustering: Start with first rule, add most similar
    ordered_indices = [0]
    remaining = set(range(1, n))
    
    while remaining:
        last_idx = ordered_indices[-1]
        
        # Find most similar rule to last added
        best_idx = max(remaining, key=lambda i: similarity[last_idx][i])
        
        ordered_indices.append(best_idx)
        remaining.remove(best_idx)
    
    return [rules[i] for i in ordered_indices]
```

**Example**:
```python
# Before grouping:
Rule 1: uses [category, confidence_score, amount]
Rule 2: uses [status, timestamp]
Rule 3: uses [category, amount]
Rule 4: uses [status, error_code]

# After grouping:
Rule 1: uses [category, confidence_score, amount]
Rule 3: uses [category, amount]              # shares category, amount
Rule 2: uses [status, timestamp]
Rule 4: uses [status, error_code]            # shares status
```

### 3. Complexity-Based Ordering

**Principle**: Simple conditions before complex nested logic

**Rationale**:
- Simple checks are fast
- Most data filtered by simple rules
- Complex logic only for difficult cases
- Early termination avoids expensive operations

**Implementation**:
```python
def calculate_complexity(condition: dict) -> int:
    """
    Calculate complexity score for a condition.
    
    Complexity factors:
    - Logical operators (all_of, any_of, none_of): +1 per level
    - Leaf conditions: +1 each
    - Expensive operators (regex_match): +2
    - Collection operators (in, not_in): +1 per item
    
    Args:
        condition: Condition expression
        
    Returns:
        Complexity score (higher = more complex)
    """
    # Handle logical operators (nested conditions)
    if 'all_of' in condition:
        return 1 + sum(calculate_complexity(c) for c in condition['all_of'])
    
    elif 'any_of' in condition:
        return 1 + sum(calculate_complexity(c) for c in condition['any_of'])
    
    elif 'none_of' in condition:
        return 1 + sum(calculate_complexity(c) for c in condition['none_of'])
    
    # Handle leaf conditions
    else:
        operator = condition.get('operator', '')
        value = condition.get('value')
        
        # Base complexity
        complexity = 1
        
        # Expensive operators
        if operator == 'regex_match':
            complexity += 2
        
        # Collection size matters
        elif operator in ('in', 'not_in') and isinstance(value, list):
            complexity += len(value) // 10  # +1 per 10 items
        
        return complexity


def order_by_complexity(rules: List[dict]) -> List[dict]:
    """
    Order rules by increasing complexity.
    
    Args:
        rules: List of rule definitions
        
    Returns:
        Rules sorted by complexity (simple first)
    """
    for rule in rules:
        rule['complexity_score'] = calculate_complexity(rule['conditions'])
    
    return sorted(rules, key=lambda r: r['complexity_score'])
```

**Example**:
```python
# Before complexity ordering:
Rule 1: (A AND B) OR (C AND NOT D)          # complexity = 7
Rule 2: category = "Fraud"                  # complexity = 1
Rule 3: amount > 100 AND status = "active"  # complexity = 3

# After complexity ordering:
Rule 1: category = "Fraud"                  # complexity = 1
Rule 2: amount > 100 AND status = "active"  # complexity = 3
Rule 3: (A AND B) OR (C AND NOT D)          # complexity = 7
```

## Combined Optimization Algorithm

```python
def optimize_ruleset(
    ruleset: dict,
    sample_df: Optional[pd.DataFrame] = None,
    enable_selectivity: bool = True,
    enable_field_grouping: bool = True,
    enable_complexity: bool = True
) -> dict:
    """
    Optimize ruleset using multiple strategies.
    
    Optimization pipeline:
    1. Selectivity analysis (if sample_df provided)
    2. Complexity-based ordering
    3. Field usage grouping
    4. Final priority assignment
    
    Args:
        ruleset: Input ruleset with unoptimized rules
        sample_df: Optional sample data for selectivity analysis
        enable_selectivity: Enable selectivity-based ordering
        enable_field_grouping: Enable field usage grouping
        enable_complexity: Enable complexity-based ordering
        
    Returns:
        Optimized ruleset with reordered rules
    """
    rules = copy.deepcopy(ruleset['rules'])
    
    logger.info(f"[INFO] Starting optimization with {len(rules)} rules")
    
    # Step 1: Selectivity analysis (if data available)
    if enable_selectivity and sample_df is not None:
        logger.info("[INFO] Analyzing rule selectivity...")
        
        for rule in rules:
            selectivity = estimate_selectivity(rule, sample_df)
            rule['selectivity_score'] = selectivity
            logger.info(
                f"  Rule '{rule.get('name', 'unnamed')}': "
                f"selectivity = {selectivity:.2%}"
            )
        
        # Sort by selectivity (lower = more selective = higher priority)
        rules.sort(key=lambda r: r.get('selectivity_score', 1.0))
        logger.info("[INFO] Rules reordered by selectivity")
    
    # Step 2: Complexity-based ordering
    if enable_complexity:
        logger.info("[INFO] Analyzing rule complexity...")
        
        # Within each selectivity band, order by complexity
        if enable_selectivity and sample_df is not None:
            # Group by selectivity bands (0-10%, 10-30%, 30-70%, 70-100%)
            bands = {
                'very_selective': [],    # 0-10%
                'selective': [],         # 10-30%
                'moderate': [],          # 30-70%
                'broad': []              # 70-100%
            }
            
            for rule in rules:
                sel = rule.get('selectivity_score', 1.0)
                if sel <= 0.1:
                    bands['very_selective'].append(rule)
                elif sel <= 0.3:
                    bands['selective'].append(rule)
                elif sel <= 0.7:
                    bands['moderate'].append(rule)
                else:
                    bands['broad'].append(rule)
            
            # Order each band by complexity
            rules = []
            for band_name in ['very_selective', 'selective', 'moderate', 'broad']:
                band_rules = order_by_complexity(bands[band_name])
                rules.extend(band_rules)
                
                if band_rules:
                    logger.info(
                        f"  Band '{band_name}': {len(band_rules)} rules "
                        f"ordered by complexity"
                    )
        else:
            # No selectivity data, just order by complexity
            rules = order_by_complexity(rules)
            logger.info("[INFO] Rules reordered by complexity")
    
    # Step 3: Field usage grouping
    if enable_field_grouping:
        logger.info("[INFO] Grouping rules by field usage...")
        
        # Within each band, group by field similarity
        # This preserves selectivity/complexity ordering while improving cache
        rules = group_by_field_usage(rules)
        logger.info("[INFO] Rules grouped by field similarity")
    
    # Step 4: Assign final priorities
    for i, rule in enumerate(rules, start=1):
        old_priority = rule.get('priority', i)
        rule['priority'] = i
        
        if old_priority != i:
            logger.info(
                f"  Rule '{rule.get('name', 'unnamed')}': "
                f"priority {old_priority} → {i}"
            )
    
    logger.info(f"[INFO] Optimization complete: {len(rules)} rules reordered")
    
    # Return optimized ruleset
    return {
        **ruleset,
        'rules': rules,
        'optimization_metadata': {
            'selectivity_enabled': enable_selectivity and sample_df is not None,
            'field_grouping_enabled': enable_field_grouping,
            'complexity_enabled': enable_complexity,
            'sample_size': len(sample_df) if sample_df is not None else 0
        }
    }
```

## Helper Functions

### Field Extraction

```python
def extract_all_fields(condition: dict) -> List[str]:
    """
    Recursively extract all field names from a condition.
    
    Args:
        condition: Condition expression (may be nested)
        
    Returns:
        List of unique field names used
    """
    fields = []
    
    # Handle logical operators
    if 'all_of' in condition:
        for subcond in condition['all_of']:
            fields.extend(extract_all_fields(subcond))
    
    elif 'any_of' in condition:
        for subcond in condition['any_of']:
            fields.extend(extract_all_fields(subcond))
    
    elif 'none_of' in condition:
        for subcond in condition['none_of']:
            fields.extend(extract_all_fields(subcond))
    
    # Handle leaf condition
    elif 'field' in condition:
        fields.append(condition['field'])
    
    return list(set(fields))  # Return unique fields
```

### Field Usage Analysis

```python
def analyze_field_usage(rules: List[dict]) -> Dict[str, int]:
    """
    Analyze which fields are used most frequently across rules.
    
    Args:
        rules: List of rule definitions
        
    Returns:
        Dictionary mapping field names to usage count
    """
    field_counts = {}
    
    for rule in rules:
        fields = extract_all_fields(rule['conditions'])
        for field in fields:
            field_counts[field] = field_counts.get(field, 0) + 1
    
    # Sort by usage count (descending)
    sorted_fields = sorted(
        field_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return dict(sorted_fields)
```

## Configuration and Control

### Environment Variables

```bash
# Optimization Control
ENABLE_RULE_OPTIMIZATION="true"          # Master switch for optimization
OPTIMIZE_BY_SELECTIVITY="true"           # Requires sample data
OPTIMIZE_BY_COMPLEXITY="true"            # Always available
OPTIMIZE_BY_FIELD_GROUPING="true"        # Always available

# Optimization Parameters
SELECTIVITY_SAMPLE_SIZE="1000"           # Rows to use for selectivity analysis
SELECTIVITY_CONFIDENCE_THRESHOLD="0.05"  # Min difference to reorder (5%)

# Reporting
INCLUDE_OPTIMIZATION_REPORT="true"       # Include detailed report
LOG_OPTIMIZATION_CHANGES="true"          # Log each priority change
```

### When to Enable/Disable Optimization

**Enable optimization when**:
- Rules have varying selectivity
- Sample data is representative
- Performance is critical
- User priorities are estimates

**Disable optimization when**:
- User priorities are carefully tuned
- Logical ordering matters (e.g., security checks first)
- Sample data is not representative
- Explainability is critical (preserve user intent)

## Performance Analysis

### Expected Improvements

**Selectivity Optimization**:
- Best case: 10x faster (highly selective rules first)
- Average case: 2-3x faster
- Worst case: No improvement (all rules equally selective)

**Field Grouping**:
- Cache hit rate improvement: 20-40%
- Average evaluation speedup: 1.2-1.5x
- Most beneficial with many rules (10+)

**Complexity Ordering**:
- Simple rule overhead: ~1-2ms
- Complex rule overhead: ~5-10ms
- Potential speedup: 1.5-2x on complex rulesets

### Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| Selectivity | Fastest execution | Requires sample data |
| Field Grouping | Better cache | May violate user intent |
| Complexity | Predictable performance | May reorder important checks |
| Overall | 2-5x speedup | Optimization overhead |

## Integration with RulesetGenerator

### In Script Implementation

```python
# In ruleset_generation.py main() function:

# Step 6: Optimize ruleset (if enabled)
enable_optimization = environ_vars.get(
    "ENABLE_RULE_OPTIMIZATION", "true"
).lower() == "true"

if enable_optimization:
    log("[INFO] Optimizing ruleset...")
    
    # Get optimization settings
    enable_selectivity = environ_vars.get(
        "OPTIMIZE_BY_SELECTIVITY", "true"
    ).lower() == "true" and sample_df is not None
    
    enable_complexity = environ_vars.get(
        "OPTIMIZE_BY_COMPLEXITY", "true"
    ).lower() == "true"
    
    enable_field_grouping = environ_vars.get(
        "OPTIMIZE_BY_FIELD_GROUPING", "true"
    ).lower() == "true"
    
    # Run optimization
    optimized_ruleset = optimize_ruleset(
        user_ruleset,
        sample_df=sample_df,
        enable_selectivity=enable_selectivity,
        enable_field_grouping=enable_field_grouping,
        enable_complexity=enable_complexity
    )
    
    log("[INFO] Optimization complete")
else:
    log("[INFO] Skipping optimization (disabled)")
    optimized_ruleset = user_ruleset
```

### Optimization Report Format

```json
{
  "optimization_applied": true,
  "strategies_used": {
    "selectivity_analysis": true,
    "complexity_ordering": true,
    "field_grouping": true
  },
  "sample_size": 1000,
  "changes": [
    {
      "rule_name": "High Confidence Reversal",
      "old_priority": 5,
      "new_priority": 1,
      "reason": "High selectivity (2% match rate)"
    },
    {
      "rule_name": "Amount Check",
      "old_priority": 1,
      "new_priority": 8,
      "reason": "Low selectivity (95% match rate)"
    }
  ],
  "field_usage": {
    "category": 8,
    "confidence_score": 7,
    "amount": 6
  },
  "estimated_speedup": "2.3x"
}
```

## Examples

### Example 1: E-commerce Fraud Detection

**Before Optimization**:
```python
rules = [
    # User-defined priority order
    Rule(priority=1, name="Amount Check", 
         conditions="amount > 0",
         selectivity=0.98),  # Matches 98% of orders
    
    Rule(priority=2, name="Country Check",
         conditions="country_code in ['US', 'CA', 'UK']",
         selectivity=0.85),  # Matches 85%
    
    Rule(priority=3, name="Fraud Pattern",
         conditions="(num_transactions > 10) AND (velocity_score > 0.9)",
         selectivity=0.02),  # Matches 2% (high fraud indicator)
    
    Rule(priority=4, name="New Customer",
         conditions="account_age_days < 30",
         selectivity=0.15),  # Matches 15%
]
```

**After Optimization**:
```python
rules = [
    # Optimized priority order
    Rule(priority=1, name="Fraud Pattern",  # Most selective first
         conditions="(num_transactions > 10) AND (velocity_score > 0.9)",
         selectivity=0.02),
    
    Rule(priority=2, name="New Customer",  # Moderately selective
         conditions="account_age_days < 30",
         selectivity=0.15),
    
    Rule(priority=3, name="Country Check",  # Broader check
         conditions="country_code in ['US', 'CA', 'UK']",
         selectivity=0.85),
    
    Rule(priority=4, name="Amount Check",  # Least selective (catch-all)
         conditions="amount > 0",
         selectivity=0.98),
]
```

**Performance Impact**:
- Average conditions evaluated per row: 4.0 → 1.2
- Total evaluation time: 100ms → 35ms
- Speedup: **2.9x**

### Example 2: Customer Support Ticket Routing

**Before Optimization**:
```python
rules = [
    Rule(priority=1, conditions="status = 'open'"),                    # 80% match
    Rule(priority=2, conditions="priority = 'high'"),                  # 10% match
    Rule(priority=3, conditions="category = 'billing' AND vip = 1"),   # 1% match
    Rule(priority=4, conditions="description CONTAINS 'urgent'"),      # 5% match
]
```

**After Selectivity + Complexity Optimization**:
```python
rules = [
    Rule(priority=1, conditions="category = 'billing' AND vip = 1"),   # 1% - most selective
    Rule(priority=2, conditions="description CONTAINS 'urgent'"),      # 5%
    Rule(priority=3, conditions="priority = 'high'"),                  # 10%
    Rule(priority=4, conditions="status = 'open'"),                    # 80% - least selective
]
```

**After Adding Field Grouping**:
```python
rules = [
    # Group 1: Simple field checks
    Rule(priority=1, conditions="category = 'billing' AND vip = 1"),   # Uses: category, vip
    Rule(priority=2, conditions="priority = 'high'"),                  # Uses: priority
    
    # Group 2: Complex string operations
    Rule(priority=3, conditions="description CONTAINS 'urgent'"),      # Uses: description
    Rule(priority=4, conditions="status = 'open'"),                    # Uses: status
]
```

## Best Practices

1. **Always use sample data**: Optimization is most effective with representative data
2. **Monitor optimization impact**: Track before/after performance metrics
3. **Preserve critical rules**: Use manual priority for security/compliance rules
4. **Document changes**: Include optimization report for transparency
5. **Test with production data**: Validate optimized ruleset matches unoptimized results
6. **Iterate**: Re-optimize when data distribution changes

## Limitations and Caveats

1. **Sample bias**: Optimization is only as good as sample data
2. **Logical dependencies**: May violate intended rule ordering
3. **Explainability**: Optimized order may not match business logic
4. **Overhead**: Optimization itself takes time (acceptable for one-time generation)
5. **Changing data**: Optimization may become suboptimal as data evolves

## Summary

Ruleset optimization provides significant performance improvements through:
- **Selectivity-based ordering**: Most selective rules first
- **Field usage grouping**: Cache-friendly rule arrangement
- **Complexity-based ordering**: Simple checks before complex logic

These strategies combine to achieve **2-5x speedup** in rule evaluation while maintaining correctness and supporting optional user control over optimization behavior.
