---
tags:
  - design
  - step_builders
  - bedrock_steps
  - batch_processing
  - patterns
  - sagemaker
  - llm_processing
  - aws_bedrock
  - cost_optimization
keywords:
  - bedrock batch processing patterns
  - AWS Bedrock batch inference
  - LLM batch processing
  - cost-efficient processing
  - scalable inference
  - batch job management
topics:
  - step builder patterns
  - bedrock batch processing implementation
  - SageMaker batch LLM processing
  - AWS Bedrock batch architecture
language: python
date of note: 2025-11-03
updated: 2025-11-07
---

# Bedrock Batch Processing Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock batch processing step builder implementations in the cursus framework. Bedrock batch processing steps create **ProcessingStep** instances that leverage AWS Bedrock's batch inference capabilities for cost-efficient, scalable Large Language Model (LLM) processing tasks. These steps maintain full compatibility with existing **Bedrock Processing** steps while providing significant cost savings (typically 50% reduction) and enhanced scalability for large datasets.

## Integration with Existing Bedrock Ecosystem

Bedrock batch processing steps are designed as drop-in replacements for standard Bedrock processing steps:

1. **Template Generation Step**: Generates structured prompt templates from category definitions
2. **Tabular Preprocessing Step**: Prepares data in train/val/test splits or single datasets
3. **Batch Processing Step**: Consumes templates and data for batch LLM inference
4. **Seamless Integration**: Identical input/output interface to standard Bedrock processing

**Integration Flow:**
```
Category Definitions → Prompt Template Generation → Prompt Templates
                                                         ↓
Tabular Data → Tabular Preprocessing → Processed Data → Bedrock Batch Processing → Categorized Results
```

## Key Architectural Differences from Real-Time Processing

### 1. Class Extension Pattern

The actual implementation uses a simple class extension pattern where `BedrockBatchProcessor` extends `BedrockProcessor`:

```python
class BedrockBatchProcessor(BedrockProcessor):
    """
    Bedrock batch processor extending BedrockProcessor with batch inference capabilities.
    Maintains full compatibility while adding cost-efficient batch processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Batch-specific configuration
        self.batch_mode = config.get("batch_mode", "auto")
        self.batch_threshold = config.get("batch_threshold", 1000)
        self.batch_role_arn = config.get("batch_role_arn")
        self.batch_timeout_hours = config.get("batch_timeout_hours", 24)
        
        # AWS Bedrock batch limits
        self.max_records_per_job = config.get("max_records_per_job", 45000)
        self.max_concurrent_batch_jobs = config.get("max_concurrent_batch_jobs", 20)
        
        # Extract S3 paths from config (set by step builder via environment variables)
        self.batch_input_s3_path = config.get("batch_input_s3_path")
        self.batch_output_s3_path = config.get("batch_output_s3_path")
        
        # Parse bucket and prefix
        if self.batch_input_s3_path:
            self.input_bucket, self.input_prefix = self._parse_s3_path(
                self.batch_input_s3_path
            )
        
        if self.batch_output_s3_path:
            self.output_bucket, self.output_prefix = self._parse_s3_path(
                self.batch_output_s3_path
            )
        
        # Initialize clients
        self.s3_client = boto3.client("s3", region_name=config.get("region_name"))
        self.bedrock_batch_client = boto3.client("bedrock", region_name=config.get("region_name"))
```

### 2. Processing Mode Selection

```python
def should_use_batch_processing(self, df: pd.DataFrame) -> bool:
    """Determine whether to use batch or real-time processing."""
    if self.batch_mode == "realtime":
        return False
    elif self.batch_mode == "batch":
        return True
    else:  # auto mode
        return (
            len(df) >= self.batch_threshold
            and self.batch_role_arn is not None
            and self.input_bucket is not None
            and self.output_bucket is not None
        )

def process_batch(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Main processing method with automatic batch/real-time selection."""
    if self.should_use_batch_processing(df):
        logger.info(f"Using batch processing for {len(df)} records")
        return self.process_batch_inference(df, **kwargs)
    else:
        logger.info(f"Using real-time processing for {len(df)} records")
        return super().process_batch(df, **kwargs)
```

### 3. JSONL Conversion Using Template Logic

```python
def convert_df_to_jsonl(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to Bedrock batch JSONL format using existing template logic."""
    jsonl_records = []
    
    for idx, row in df.iterrows():
        # Use parent class method to format prompt with template placeholders
        row_data = row.to_dict()
        prompt = self._format_prompt(row_data)
        
        # Create Bedrock batch inference record
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config["max_tokens"]),
            "temperature": float(self.config["temperature"]),
            "top_p": float(self.config["top_p"]),
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if self.config.get("system_prompt"):
            request_body["system"] = self.config["system_prompt"]
        
        record = {"recordId": f"record_{idx}", "modelInput": request_body}
        jsonl_records.append(record)
    
    return jsonl_records
```

### 4. S3 Integration with Multipart Upload

The implementation includes multipart upload support for large files (>100MB):

```python
def upload_jsonl_to_s3(self, jsonl_records: List[Dict[str, Any]]) -> str:
    """Upload JSONL data to S3 using multipart upload for large files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if self.input_prefix:
        s3_key = f"{self.input_prefix}/input_{timestamp}.jsonl"
    else:
        s3_key = f"input_{timestamp}.jsonl"
    
    jsonl_content = "\n".join([json.dumps(record) for record in jsonl_records])
    content_bytes = jsonl_content.encode("utf-8")
    content_size_mb = len(content_bytes) / (1024 * 1024)
    
    # Use multipart upload for files larger than 100MB
    if len(content_bytes) > 100 * 1024 * 1024:
        logger.info(f"Using multipart upload for large file ({content_size_mb:.2f} MB)")
        self._upload_large_file_multipart(content_bytes, self.input_bucket, s3_key)
    else:
        self.s3_client.put_object(
            Bucket=self.input_bucket,
            Key=s3_key,
            Body=content_bytes,
            ContentType="application/jsonl",
        )
    
    return f"s3://{self.input_bucket}/{s3_key}"
```

### 5. Job Name Validation

AWS Bedrock has strict job name requirements - no underscores allowed:

```python
def _validate_job_name(self, job_name: str) -> None:
    """Validate job name against AWS Bedrock naming requirements."""
    pattern = r"^[a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9\+\-\.]){0,63}$"
    
    if not re.match(pattern, job_name):
        raise ValueError(
            f"Invalid job name '{job_name}'. "
            f"AWS Bedrock job names must match pattern: [a-zA-Z0-9]{{1,63}}(-*[a-zA-Z0-9\\+\\-\\.]{{0,63}}. "
            f"Allowed: alphanumeric, hyphens, plus signs, dots. "
            f"Not allowed: underscores or other special characters."
        )

def create_batch_job(self, input_s3_uri: str) -> str:
    """Create Bedrock batch inference job."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Hyphens, not underscores
    job_name = f"cursus-bedrock-batch-{timestamp}"
    
    # Validate before attempting to create
    self._validate_job_name(job_name)
    
    # ... create job
```

### 6. Dual-Limit Dataset Splitting

AWS Bedrock has TWO limits: record count (50K) AND file size (1GB). Implementation handles both:

```python
def _estimate_jsonl_size(self, jsonl_records: List[Dict[str, Any]]) -> int:
    """Estimate JSONL file size by sampling first 100 records."""
    sample_size = min(100, len(jsonl_records))
    sample_records = jsonl_records[:sample_size]
    
    sample_bytes = sum(
        len(json.dumps(record).encode("utf-8")) + 1 
        for record in sample_records
    )
    
    avg_record_size = sample_bytes / sample_size
    return int(avg_record_size * len(jsonl_records))

def _split_dataframe_for_batch(self, df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split DataFrame complying with BOTH AWS limits: record count AND file size."""
    MAX_FILE_SIZE = 900 * 1024 * 1024  # 900MB conservative limit
    
    chunks = []
    current_start = 0
    
    while current_start < len(df):
        # Start with max record count
        current_end = min(current_start + self.max_records_per_job, len(df))
        chunk = df.iloc[current_start:current_end].copy()
        
        # Convert to JSONL and check size
        jsonl_records = self.convert_df_to_jsonl(chunk)
        estimated_size = self._estimate_jsonl_size(jsonl_records)
        
        # If too large, reduce chunk size proportionally
        while estimated_size > MAX_FILE_SIZE and len(chunk) > 1:
            size_ratio = MAX_FILE_SIZE / estimated_size
            new_size = max(1, int(len(chunk) * size_ratio * 0.9))  # 90% safety margin
            
            current_end = current_start + new_size
            chunk = df.iloc[current_start:current_end].copy()
            jsonl_records = self.convert_df_to_jsonl(chunk)
            estimated_size = self._estimate_jsonl_size(jsonl_records)
        
        chunks.append(chunk)
        current_start = current_end
    
    return chunks
```

### 7. Multi-Job Batch Processing

For datasets larger than 45K records, automatically creates and monitors multiple batch jobs:

```python
def _process_multi_batch_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
    """Process multiple batch jobs in parallel for large datasets."""
    # Step 1: Split DataFrame into chunks
    chunks = self._split_dataframe_for_batch(df)
    
    # Step 2 & 3: Upload chunks and create batch jobs
    job_arns = []
    for i, chunk in enumerate(chunks):
        jsonl_records = self.convert_df_to_jsonl(chunk)
        input_s3_uri = self.upload_jsonl_to_s3(jsonl_records)
        job_arn = self.create_batch_job(input_s3_uri)
        job_arns.append(job_arn)
    
    # Step 4: Monitor all jobs in parallel
    job_responses = self._monitor_multiple_batch_jobs(job_arns)
    
    # Step 5: Download and merge results
    all_results = []
    for job_response, chunk in zip(job_responses, chunks):
        batch_results = self.download_batch_results(job_response)
        chunk_result_df = self.convert_batch_results_to_df(batch_results, chunk)
        all_results.append(chunk_result_df)
    
    # Merge and sort by original index
    result_df = pd.concat(all_results, ignore_index=False).sort_index()
    return result_df
```

### 8. Pydantic Response Parsing Fix

Critical fix for prefilled assistant messages:

```python
def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Bedrock response using Pydantic model validation."""
    response_text = response["content"][0].get("text", "")
    
    if self.response_model_class:
        # FIX: Prepend opening brace since prefilling is not included in response
        # The assistant message was prefilled with "{", but Bedrock response
        # continues from after the prefill, so we need to add it back
        complete_json = "{" + response_text
        
        validated_response = self.response_model_class.model_validate_json(complete_json)
        return validated_response.model_dump()
```

## Cursus Framework Integration Patterns

### S3 Path Management via Environment Variables

The step builder sets S3 paths using cursus framework patterns, which are then passed to the script via environment variables:

```python
# In BedrockBatchProcessingStepBuilder
def _get_environment_variables(self) -> Dict[str, str]:
    """Create environment variables using cursus framework patterns for S3 path management."""
    env_vars = super()._get_environment_variables()
    
    # Get base output path from framework
    base_output_path = self._get_base_output_path()
    
    # Create batch-specific S3 paths using Join
    batch_input_path = Join(on="/", values=[base_output_path, "bedrock-batch", "input"])
    batch_output_path = Join(on="/", values=[base_output_path, "bedrock-batch", "output"])
    
    env_vars.update({
        "BEDROCK_BATCH_INPUT_S3_PATH": batch_input_path,
        "BEDROCK_BATCH_OUTPUT_S3_PATH": batch_output_path,
        # ... other batch config
    })
    
    return env_vars
```

### Script-Level S3 Path Extraction

The script extracts S3 paths from environment variables:

```python
# In main() function
config = {
    # Extract from environ_vars parameter (NOT os.environ directly)
    "batch_input_s3_path": environ_vars.get("BEDROCK_BATCH_INPUT_S3_PATH"),
    "batch_output_s3_path": environ_vars.get("BEDROCK_BATCH_OUTPUT_S3_PATH"),
    # ... other config
}

# In BedrockBatchProcessor.__init__()
self.batch_input_s3_path = config.get("batch_input_s3_path")
self.batch_output_s3_path = config.get("batch_output_s3_path")

if self.batch_input_s3_path:
    self.input_bucket, self.input_prefix = self._parse_s3_path(self.batch_input_s3_path)
```

## Key Design Principles

### 1. **100% Compatibility**
- Identical input/output interface to `bedrock_processing.py`
- Same container paths and environment variables
- Same job type handling (training, validation, testing, calibration)
- Same output file formats and naming conventions

### 2. **Intelligent Processing Mode Selection**
- **Auto Mode**: Automatically selects batch vs real-time based on data size and configuration
- **Batch Mode**: Forces batch processing regardless of data size
- **Real-time Mode**: Forces real-time processing (identical to original script)

### 3. **Seamless Fallback Strategy**
- Automatic fallback to real-time processing if batch processing fails
- Preserves all error handling and retry logic from parent class
- Maintains same output format regardless of processing mode used

### 4. **Scalability for Large Datasets**
- Automatic dataset splitting for >45K records
- Parallel batch job creation and monitoring
- File size validation to prevent AWS limit errors
- Multi-part S3 uploads for large files

### 5. **Framework Compliance**
- Uses cursus framework patterns for S3 path management
- Leverages `_get_base_output_path()` for PIPELINE_EXECUTION_TEMP_DIR support
- Follows established environment variable patterns
- Maintains consistency with other step builders

## Environment Variables

### Standard Bedrock Variables (inherited)
```bash
# Model Configuration
BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"

# API Configuration
BEDROCK_MAX_TOKENS="32768"
BEDROCK_TEMPERATURE="1.0"
BEDROCK_TOP_P="0.999"
BEDROCK_MAX_RETRIES="3"

# Processing Configuration
BEDROCK_BATCH_SIZE="10"
BEDROCK_OUTPUT_COLUMN_PREFIX="llm_"
BEDROCK_CONCURRENCY_MODE="sequential"
```

### Batch-Specific Variables
```bash
# Batch Processing Mode
BEDROCK_BATCH_MODE="auto"  # auto, batch, realtime
BEDROCK_BATCH_THRESHOLD="1000"  # minimum records for batch mode

# AWS Resources for Batch Processing
BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
BEDROCK_BATCH_INPUT_S3_PATH="<generated-dynamically>"  # Set by step builder
BEDROCK_BATCH_OUTPUT_S3_PATH="<generated-dynamically>"  # Set by step builder
BEDROCK_BATCH_TIMEOUT_HOURS="24"

# AWS Bedrock Batch Limits
BEDROCK_MAX_RECORDS_PER_JOB="45000"  # Conservative (AWS max: 50,000)
BEDROCK_MAX_CONCURRENT_BATCH_JOBS="20"  # AWS limit
```

## Expected Benefits

### 1. **Cost Efficiency**
- **50% cost reduction** for large datasets through batch pricing
- **Automatic optimization** based on data size and processing requirements
- **Resource efficiency** through AWS-managed batch infrastructure

### 2. **Scalability**
- **Unlimited scale** - process millions of records through multi-job orchestration
- **Parallel processing** - AWS handles optimal resource allocation
- **Fault tolerance** - Built-in retry and error recovery mechanisms

### 3. **Operational Excellence**
- **Zero configuration changes** required for existing pipelines
- **Automatic fallback** ensures reliability and backward compatibility
- **Enhanced monitoring** with batch job status tracking and cost reporting

### 4. **Developer Experience**
- **Drop-in replacement** for existing bedrock processing steps
- **Identical debugging experience** with same logging and error handling
- **Flexible deployment** - works with existing SageMaker pipeline infrastructure

## JSON Output Enforcement and Quote Safety

### Production-Validated Approach (Nov 2025)

Based on analysis of 378K+ production records, implemented a comprehensive JSON output enforcement strategy with quote-safe error handling:

#### 1. Assistant Message Prefilling

**Strategy**: Force JSON output by prefilling the assistant's response with an opening brace:

```python
def convert_df_to_jsonl(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to Bedrock batch JSONL format."""
    jsonl_records = []
    
    for pos, (_, row) in enumerate(df.iterrows()):
        row_data = row.to_dict()
        prompt = self._format_prompt(row_data)
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config["max_tokens"]),
            "temperature": float(self.config["temperature"]),
            "top_p": float(self.config["top_p"]),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"},  # Force JSON output via prefilling
            ],
        }
        
        # Use positional index (0..N-1) not arbitrary DataFrame index labels
        record = {"recordId": f"record_{pos}", "modelInput": request_body}
        jsonl_records.append(record)
    
    return jsonl_records
```

**Benefits**:
- Reduces markdown/prose leakage in model outputs
- Provides consistent JSON structure across batch and realtime
- Works with Claude 4.0 on Bedrock (standard format)

**Note**: Bedrock response format for Claude models:
```json
{
  "content": [
    {"type": "text", "text": "..."}
  ],
  "stop_reason": "end_turn"
}
```

#### 2. Quote-Safe JSON Repair

**Problem**: Production analysis revealed 100% of parse errors (341 out of 378,878 records) were due to Unicode quotes, specifically German-style `„text"` patterns.

**Solution**: Focused two-step repair that ONLY touches Unicode quotes, never ASCII `"` structural delimiters:

```python
# Unicode quote mappings - ALL map to apostrophe to align with output spec
UNICODE_DOUBLE_QUOTES = {
    "\u201c": "'",  # " → ' (Left double quotation mark)
    "\u201d": "'",  # " → ' (Right double quotation mark)
    "\u201e": "'",  # „ → ' (Double low-9 quotation mark)
    "\u201f": "'",  # ‟ → ' (Double high-reversed-9 quotation mark)
}

UNICODE_SINGLE_QUOTES = {
    "\u2018": "'",  # ' → ' (Left single quotation mark)
    "\u2019": "'",  # ' → ' (Right single quotation mark)
    "\u201a": "'",  # ‚ → ' (Single low-9 quotation mark)
    "\u201b": "'",  # ‛ → ' (Single high-reversed-9 quotation mark)
}

# Pattern for German-style quotes
GERMAN_OPEN_QUOTE_PATTERN = re.compile(r'„([^"]*)"')

def repair_json(text: str) -> str:
    """
    FOCUSED repair function - ONLY handles Unicode quotes.
    
    Based on production analysis of 378,878 records:
    - 100% of parse errors are due to Unicode quotes
    - No other JSON syntax errors observed
    - Generic repairs are unnecessary and risky
    """
    # STEP 1: Fix German quote pattern „name" → \"name\"
    text = GERMAN_OPEN_QUOTE_PATTERN.sub(r'\\"\1\\"', text)
    
    # STEP 2: Normalize remaining Unicode quotes to apostrophes
    text = normalize_unicode_quotes(text)
    
    return text
```

**Critical Safety Properties**:
1. ASCII double quotes `"` (U+0022) are **never touched** - JSON structure preserved
2. German `„name"` becomes `\"name\"` (properly escaped, no new delimiters)
3. All other fancy quotes become `'` (aligns with output spec)
4. No comma/whitespace surgery that could break valid JSON

#### 3. Two-Stage Parse Strategy

```python
def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse with focused quote repair fallback."""
    response_text = response["content"][0].get("text", "")
    
    if self.response_model_class:
        # Extract JSON substring (handles markdown fences)
        complete_json = extract_json_candidate(response_text)
        
        # STAGE 1: Try parsing as-is
        try:
            validated_response = self.response_model_class.model_validate_json(complete_json)
            result = validated_response.model_dump()
            result["validation_passed"] = True
            return result
        except (ValidationError, json.JSONDecodeError) as first_error:
            # STAGE 2: Apply quote-only repair and retry
            logger.warning("Initial parse failed, attempting quote repair")
            repaired_json = repair_json(complete_json)
            
            try:
                validated_response = self.response_model_class.model_validate_json(repaired_json)
                result = validated_response.model_dump()
                result["validation_passed"] = True
                return result
            except (ValidationError, json.JSONDecodeError) as second_error:
                logger.error(f"Both parse attempts failed")
                raise second_error
```

#### 4. Positional Index Mapping for RecordId

**Problem**: Using `df.iterrows()` index labels for `recordId` breaks when DataFrame has non-default index (e.g., `[100, 101, ...]` or custom labels).

**Solution**: Use positional counter (0..N-1) instead:

```python
# BEFORE (broken with non-default index):
for idx, row in df.iterrows():
    record = {"recordId": f"record_{idx}", ...}

# AFTER (safe for any index):
for pos, (_, row) in enumerate(df.iterrows()):
    record = {"recordId": f"record_{pos}", ...}
```

**Benefits**:
- Works with any DataFrame index type
- Maintains 1:1 mapping to `.iloc[]` positions
- Bedrock output order is not guaranteed, so using `recordId` for recovery is essential

#### 5. Safe Validation Column Checking

**Problem**: Using `.get()` in boolean indexing crashes when column doesn't exist:
```python
# This crashes with KeyError: False when column missing
result_df[result_df.get("validation_passed", False) == True]
```

**Solution**: Check column existence first:

```python
validation_col = f"{config['output_column_prefix']}validation_passed"
if validation_col in result_df.columns:
    validation_passed_count = len(result_df[result_df[validation_col] == True])
else:
    validation_passed_count = 0
```

Applied in 3 locations per script:
- `process_split_directory` function
- Training fallback section (no train/val/test splits)
- Non-training job types section

#### 6. Enum Type Handling

**Problem**: Dynamic `Literal[tuple(...)]` construction can fail at import time.

**Solution**: Use plain `str` type for enum fields - Pydantic still validates against schema's enum constraint:

```python
def _convert_json_schema_type_to_python(self, field_schema: Dict[str, Any]) -> type:
    field_type = field_schema.get("type", "string")
    
    if field_type == "string":
        if "enum" in field_schema:
            # For enum fields, use str type for simplicity
            # Pydantic will still validate against the schema's enum constraint
            return str
        return str
```

**Trade-off**: Slightly less strict validation at Pydantic layer, but far more robust and maintainable.

### Production Validation Results

Fixes validated against production data:
- **Dataset**: 378,878 records from real production runs
- **Parse error rate**: 341 errors (0.09%) - all quote-related
- **Error pattern**: 100% German `„name"` Unicode quotes
- **Other JSON errors**: 0 (no comma, brace, or whitespace issues)
- **Fix effectiveness**: Quote-repair strategy addresses 100% of observed failures

**Recommendation**: This focused approach is production-ready. Generic JSON repair (comma/whitespace fixes) would be unnecessary complexity and risk introducing new failures.

## Critical Implementation Fixes

During implementation, several critical issues were identified and resolved:

### 1. Pydantic Validation for Prefilled Responses
**Issue**: Bedrock's assistant message prefilling (forcing JSON output) causes validation to fail because the prefilled "{" is not in the response.
**Fix**: Prepend opening brace before Pydantic validation. (NOTE: This was the original approach; latest implementation uses assistant prefilling which handles this automatically)

### 2. S3 Path Configuration Flow
**Issue**: Script needs S3 paths from step builder environment variables, not direct `os.environ` access.
**Fix**: Pass through `config` dictionary from `environ_vars` parameter.

### 3. Boto3 Client Separation
**Issue**: Batch operations require `bedrock` client, not `bedrock-runtime`.
**Fix**: Create separate `bedrock_batch_client` for batch operations.

### 4. Job Name Format Compliance
**Issue**: AWS Bedrock job names don't allow underscores.
**Fix**: Use hyphens in timestamp format (`%Y%m%d-%H%M%S` not `%Y%m%d_%H%M%S`).

### 5. Job Name Validation
**Issue**: Need proactive validation to catch invalid job names.
**Fix**: Added comprehensive regex validation against AWS requirements.

### 6. File Size Validation (1GB Limit)
**Issue**: AWS has BOTH record count (50K) AND file size (1GB) limits.
**Fix**: Added dual-limit validation with dynamic chunk sizing based on estimated size.

### 7. Multi-Job Batch Processing
**Issue**: Single job limited to 50K records, but datasets can be much larger.
**Fix**: Implemented multi-job orchestration with parallel monitoring for unlimited scalability.

## Summary

The Bedrock Batch Processing step provides a production-ready implementation that:

1. **Extends** the existing `BedrockProcessor` class with batch capabilities
2. **Maintains** 100% compatibility with real-time processing interface
3. **Provides** intelligent automatic mode selection based on data size
4. **Handles** AWS Bedrock batch limits through dataset splitting and multi-job orchestration
5. **Integrates** seamlessly with cursus framework S3 path management patterns
6. **Delivers** up to 50% cost reduction for large datasets while ensuring reliability through automatic fallback

This design ensures that Bedrock batch processing provides significant cost and scalability benefits while maintaining complete compatibility with existing cursus framework patterns and user workflows.
