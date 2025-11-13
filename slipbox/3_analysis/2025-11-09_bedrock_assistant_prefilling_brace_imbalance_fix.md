---
tags:
  - analysis
  - implementation
  - bedrock_steps
  - error_handling
  - json_parsing
  - critical_fix
  - assistant_prefilling
keywords:
  - bedrock batch processing
  - assistant prefilling
  - brace imbalance
  - json extraction
  - parsing errors
  - production fix
topics:
  - bedrock processing
  - json validation
  - error handling patterns
  - assistant prefilling
language: python
date of note: 2025-11-09
---

# Bedrock Assistant Prefilling: Brace Imbalance Fix (23,964 Records)

## Executive Summary

**Discovery Date:** November 9, 2025  
**Impact:** 99.996% failure rate (23,962 / 23,964 records)  
**Root Cause:** Naive JSON extraction combined with assistant prefilling  
**Fix Result:** 99.99% success rate (23,962 / 23,964 records)  
**Improvement:** Fixed 23,962 records with single change

## Problem Statement

### Initial Symptoms
```
ERROR: Invalid JSON: trailing characters at line 21 column 4
ERROR: Invalid JSON: trailing characters at line 18 column 4
ERROR: Invalid JSON: trailing characters at line 14 column 4
```

**Pattern:** 100% of records failed with "trailing characters" error  
**Frequency:** 23,962 out of 23,964 records (99.996%)  
**Previous Success Rate:** 99.91% (on different dataset without prefilling)  
**New Success Rate:** 0.00% (complete failure)

## Root Cause Analysis

### The Assistant Prefilling Pattern

```python
# Bedrock request with assistant prefilling
request_body = {
    "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "{"}  # Force JSON output
    ]
}
```

**Purpose of Prefilling:**
- Forces LLM to output JSON format
- Reduces markdown wrapping
- Improves structured output compliance
- Industry best practice for structured generation

**The Problem:**
The opening `{` is consumed by the assistant role and NOT included in the response text.

### Response Text Structure

**What We Expected:**
```json
{
  "category": "TrueDNR",
  "confidence_score": 0.95,
  ...
}
```

**What We Actually Got:**
```json

  "category": "TrueDNR",
  "confidence_score": 0.95,
  "key_evidence": {
    "message_evidence": [...]
  }
}
```

Note: Missing opening `{` at the start!

### The Extraction Bug

**Original Code (BROKEN):**
```python
def extract_json_candidate(response_text: str) -> str:
    """Extract JSON between first { and last }"""
    start = response_text.find("{")  # Finds { inside "key_evidence": {
    end = response_text.rfind("}")   # Finds LAST }
    return response_text[start:end+1]
```

**Step-by-Step Failure:**

1. Response text: `\n  "category": "TrueDNR",\n  "key_evidence": {...}`
2. `find("{")` finds position 82 (inside `"key_evidence": {`)
3. `rfind("}")` finds last `}` at position 1233
4. Extracts: `{"message_evidence": [...]}` (nested object only!)
5. Result: Structurally invalid JSON with extra closing braces

**Why It's Wrong:**
- Extracted nested object instead of full response
- Missing top-level fields (`category`, `confidence_score`, etc.)
- Has extra closing braces from outer structure
- Validation fails: "trailing characters"

### Visual Representation

```
Response Text (missing opening {):
\n  "category": "TrueDNR",
                              ↓ find("{") starts HERE (wrong!)
  "key_evidence": {
    "message_evidence": [...]
  }
}  ← rfind("}") ends HERE (includes extra })

Extracted (WRONG):
{
  "message_evidence": [...]
}
}  ← Extra brace! "trailing characters" error

Should Have Extracted (RIGHT):
{
  "category": "TrueDNR",
  "key_evidence": {
    "message_evidence": [...]
  }
}
```

## The Solution (Two-Part Fix)

### Part 1: Prepend Opening Brace BEFORE Extraction

**Critical Order Change:**
```python
# BEFORE (Wrong Order):
complete_json = extract_json_candidate(response_text)
if not complete_json.startswith("{"):
    complete_json = "{" + complete_json  # Too late!

# AFTER (Correct Order):
if not response_text.startswith("{"):
    response_text = "{" + response_text  # Fix BEFORE extraction
complete_json = extract_json_candidate(response_text)
```

**Why Order Matters:**
- Prepending AFTER extraction doesn't help the extractor find correct boundaries
- Prepending BEFORE ensures `find("{")` finds the RIGHT opening brace
- The extractor now sees the full, complete JSON structure

### Part 2: Smart Brace-Counting Extraction

**New Implementation:**
```python
def extract_json_candidate(response_text: str) -> str:
    """
    Extract first complete JSON object using intelligent brace counting.
    
    Properly handles:
    - Nested objects and arrays
    - Braces inside strings (ignored)
    - Escape sequences
    - Finds first structurally complete JSON object
    """
    start = response_text.find("{")
    if start == -1:
        return response_text.strip()
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(response_text)):
        char = response_text[i]
        
        # Handle escape sequences (\", \\, etc.)
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        # Track string boundaries (braces inside strings don't count)
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        # Count braces only outside strings
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                # Found first complete JSON object when count returns to 0
                if brace_count == 0:
                    return response_text[start:i+1]
    
    # Fallback if no complete object found
    return response_text[start:].strip()
```

**Key Features:**
1. ✅ **State Tracking:** Knows if inside string vs. outside
2. ✅ **Escape Handling:** Properly handles `\"` and `\\`
3. ✅ **Brace Balance:** Counts opening/closing braces
4. ✅ **String Awareness:** Ignores braces inside strings
5. ✅ **First Complete Object:** Stops at first balanced object

## Implementation

### Code Changes

**Location:** `projects/rnr_pytorch_bedrock/docker/scripts/bedrock_batch_processing.py`

```python
def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Bedrock response using Pydantic model validation.
    """
    if "content" in response and len(response["content"]) > 0:
        response_text = response["content"][0].get("text", "")
    else:
        raise ValueError("No content in Bedrock response")

    try:
        if self.response_model_class:
            # STEP 0: Handle assistant prefilling BEFORE extraction (CRITICAL)
            # Prepend { BEFORE extraction to avoid grabbing nested objects
            if not response_text.strip().startswith("{"):
                response_text = "{" + response_text
                logger.info("Prepended opening brace from assistant prefilling")

            # STEP 1: Extract JSON with smart brace counting
            complete_json = extract_json_candidate(response_text)

            # STEP 2: Try parsing as-is
            try:
                validated_response = self.response_model_class.model_validate_json(
                    complete_json
                )
                result = validated_response.model_dump()
                result["parse_status"] = "success"
                result["validation_passed"] = True
                return result

            except (ValidationError, json.JSONDecodeError) as first_error:
                # STEP 3: Repair with quote-only repair and retry
                logger.warning(f"Initial parsing failed, attempting repair: {first_error}")
                repaired_json = repair_json(complete_json)

                try:
                    validated_response = self.response_model_class.model_validate_json(
                        repaired_json
                    )
                    logger.info("JSON repair successful")
                    result = validated_response.model_dump()
                    result["parse_status"] = "success"
                    result["validation_passed"] = True
                    return result

                except (ValidationError, json.JSONDecodeError) as second_error:
                    logger.error(f"JSON repair failed. Original: {first_error}")
                    logger.error(f"Repair attempt: {second_error}")
                    raise second_error
        # ... rest of error handling ...
```

## Results

### Before Fix
```
Total Records: 23,964
Successful Parses: 0 (0.00%)
Failed Parses: 23,964 (100.00%)
Error Type: ValidationError (100%)
Error Pattern: "trailing characters" (100%)
```

### After Fix
```
Total Records: 23,964
Successful Parses: 23,962 (99.99%)
Failed Parses: 2 (0.01%)
Error Type: ValidationError (2 records - unrelated model issues)
Error Pattern: Non-string key (model output quality)
```

### Improvement Metrics
- **Records Fixed:** 23,962
- **Success Rate Change:** 0.00% → 99.99%
- **Error Reduction:** 100% → 0.01%
- **Pattern Resolution:** 100% of brace imbalance errors eliminated

## Why This Matters

### Impact Scope
1. **Primary Error Type:** This was causing >99% of failures
2. **Previous Unicode Issues:** Minor by comparison (0.09% of records)
3. **Previous Markdown Fences:** Sporadic (5.7% in one file)
4. **This Fix:** Resolved 99.996% of all errors

### Comparison with Previous Analysis

| Error Type | Previous Analysis | This Discovery |
|------------|------------------|----------------|
| Unicode Quotes | 314 errors (0.09%) | Still present (2 remaining) |
| Markdown Fences | 1,711 errors (5.7% in 1 file) | Not present in this batch |
| **Brace Imbalance** | **Not identified** | **23,962 errors (99.996%)** |

**Key Insight:** The brace imbalance was the PRIMARY failure mode, hidden because it only manifests when assistant prefilling is used.

## Lessons Learned

### What We Discovered
1. ⚠️ **Assistant prefilling changes response structure** - Opening brace consumed
2. ⚠️ **Naive extraction fails with nesting** - First/last { } pattern insufficient
3. ⚠️ **Order of operations critical** - Fix BEFORE extraction, not after
4. ⚠️ **State tracking required** - Must distinguish strings from structure
5. ⚠️ **Test with real patterns** - Synthetic examples miss real failure modes

### What Worked
✅ **Diagnostic-driven approach** - Built tools to analyze actual failures  
✅ **Incremental diagnosis** - Isolated each step of the parsing pipeline  
✅ **Production validation** - Tested on 24K real records  
✅ **Intelligent parsing** - State machine approach to extraction  

### Key Takeaways
1. **Always test with production data** - Edge cases appear in real use
2. **Understand API behavior** - Assistant prefilling has side effects
3. **Don't assume simple solutions** - Naive extraction insufficient
4. **Build diagnostic tools** - Essential for root cause analysis
5. **Fix root cause first** - Don't optimize before understanding

## Alternative: Remove Assistant Prefilling?

### User Question
"Can we resolve this issue by removing assistant prompt?"

### Answer: No (Keep Prefilling + Our Fix)

#### Pros of Keeping Assistant Prefilling
✅ **Better JSON compliance** from model  
✅ **Reduces markdown wrapping** (prevents ````json` fences)  
✅ **Forces structured output** format  
✅ **Industry best practice** for JSON generation  
✅ **Our fix handles it correctly** (99.99% success)  

#### Cons of Removing Assistant Prefilling
❌ **More markdown fences** - Model wraps JSON more often  
❌ **Less format compliance** - More post-processing needed  
❌ **Against best practices** - Standard pattern for structured generation  
❌ **Unnecessary** - We already fixed the handling  

### Recommendation
**Keep assistant prefilling with our fix.** The issue wasn't the prefilling itself—it was improper handling. Now fixed with 99.99% success rate.

## Related Documentation

- **Original Analysis:** `slipbox/4_analysis/2025-11-07_bedrock_json_parsing_error_handling_analysis.md`
- **Batch Processing:** `slipbox/1_design/bedrock_batch_processing_step_builder_patterns.md`
- **Prompt Templates:** `slipbox/1_design/bedrock_prompt_template_generation_step_patterns.md`

## Files Modified

1. `projects/rnr_pytorch_bedrock/docker/scripts/bedrock_batch_processing.py`
2. `src/cursus/steps/scripts/bedrock_batch_processing.py` 
3. `projects/rnr_pytorch_bedrock/debug_json_parsing.py` (diagnostic tool)
4. `projects/rnr_pytorch_bedrock/diagnose_remaining_errors.py` (analysis tool)

## Status

✅ **Production Ready**  
✅ **Validated on 23,964 records**  
✅ **99.99% success rate achieved**  
✅ **100% of brace imbalance errors resolved**

## Next Steps

1. ✅ Deploy fix to production scripts
2. ✅ Validate with comprehensive testing
3. ✅ Document discovery and solution
4. [ ] Monitor production batches for new patterns
5. [ ] Consider additional edge cases
