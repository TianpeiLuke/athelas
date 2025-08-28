---
tags:
  - design
  - documentation
  - standardization
  - metadata
keywords:
  - YAML frontmatter
  - documentation standard
  - metadata format
  - knowledge organization
  - searchability
topics:
  - documentation standards
  - knowledge management
  - organizational structure
language: python
date of note: 2025-07-31
---

# Documentation YAML Frontmatter Standard

## Purpose

This document defines the standardized YAML frontmatter format for all documentation files in the MODS project. The frontmatter provides consistent metadata that improves discoverability, categorization, and relationships between documents.

## Frontmatter Schema

All documentation files should include the following YAML frontmatter at the beginning of the file:

```yaml
---
tags:
  - tag1
  - tag2
  - tag3
keywords:
  - keyword1
  - keyword2
  - keyword3
topics:
  - topic1
  - topic2
language: python
date of note: YYYY-MM-DD
---
```

### Field Definitions

1. **tags**: Broad categorization for document filtering and navigation
   - Should include 3-5 hierarchical tags moving from general to specific
   - First tag typically identifies document type (code, project, design, analysis)
   - Following tags provide increasingly specific categorization

2. **keywords**: Specific terms for search and indexing
   - Should include 3-10 relevant terms
   - Each term should be descriptive and contextually relevant
   - Terms should help with document discovery

3. **topics**: Major subject areas covered in the document
   - Should include 2-4 main topics
   - Topics are broader than keywords but more specific than tags
   - Help cluster related documents by subject area

4. **language**: Primary programming language discussed
   - Usually "python" for our codebase
   - Can be another language if specifically relevant

5. **date of note**: Creation/last major update date
   - Format: YYYY-MM-DD
   - Typically extracted from the filename if using date-based naming

## Special Tag Types

### Entry Point Tag

Documents that serve as entry points or main indexes for a module should include `entry_point` as the first tag:

```yaml
tags:
  - entry_point
  - code
  - [module_name]
  - documentation
```

Example locations for entry point tags:
- Module README.md files
- Index documents
- Getting started guides

### Code Tags

Documentation for code components should include the module path:

```yaml
tags:
  - code
  - pipeline_api
  - [component_name]
  - [component_purpose]
```

### Design Tags

Documentation for design documents should begin with "design":

```yaml
tags:
  - design
  - [design_area]
  - [component_name]
  - [architecture_aspect]
  - [design_purpose]
```

Example locations for design tags:
- Architecture design documents
- System design specifications
- Component design documents
- Technical design proposals

### Project Tags

Project documentation should categorize planning, implementation or analysis:

```yaml
tags:
  - project
  - [planning/implementation/analysis]
  - [technical_area]
  - [component]
```

### Test Tags

Documentation for test components should begin with "test":

```yaml
tags:
  - test
  - [test_type]
  - [component_name]
  - [test_purpose]
```

Example locations for test tags:
- Test suite documentation
- Test case specifications
- Validation framework documentation
- Test runner guides
- Testing methodology documents

Common test type subcategories:
- `unit` - Unit test documentation
- `integration` - Integration test documentation
- `validation` - Validation framework documentation
- `builders` - Step builder test documentation
- `end_to_end` - End-to-end test documentation
- `performance` - Performance test documentation

## Examples

### Module README Example (Entry Point)

```yaml
---
tags:
  - entry_point
  - code
  - pipeline_api
  - documentation
  - overview
keywords:
  - pipeline API
  - DAG
  - template converter
  - MODS integration
  - documentation
topics:
  - pipeline API
  - usage examples
  - architecture
language: python
date of note: 2025-07-31
---
```

### Component Documentation Example

```yaml
---
tags:
  - code
  - pipeline_api
  - config_resolver
  - matching_engine
keywords:
  - configuration
  - resolver
  - matching
  - DAG
  - pipeline
topics:
  - pipeline API
  - configuration resolution
language: python
date of note: 2025-07-31
---
```

### Project Planning Document Example

```yaml
---
tags:
  - project
  - planning
  - validation
  - alignment
keywords:
  - alignment validation
  - script contracts
  - property paths
  - step specifications
  - validation framework
topics:
  - pipeline validation
  - contract alignment
  - property path consistency
  - implementation plan
language: python
date of note: 2025-07-05
---
```

### Test Documentation Example

```yaml
---
tags:
  - test
  - builders
  - tabular_preprocessing
  - validation
keywords:
  - step builder tests
  - tabular preprocessing
  - validation infrastructure
  - universal test framework
  - processing step validation
topics:
  - test suite documentation
  - validation framework
  - step builder testing
language: python
date of note: 2025-08-08
---
```

## Implementation Process

The YAML frontmatter standardization was implemented across all documentation in:

1. `slipbox/project_planning/` - Project planning and implementation documents
2. `slipbox/pipeline_api/` - API component documentation
3. `slipbox/pipeline_design/` - Design documents and architecture

The implementation process included:
1. Analyzing existing documentation for common themes
2. Creating a standardized schema for metadata
3. Adding appropriate tags, keywords, and topics based on document content
4. Ensuring consistent formatting across all documents
5. Special handling for entry point documents and READMEs

## Benefits

### 1. Improved Discoverability
- Documents can be found via tags, keywords, or topics
- Related documents can be discovered through shared metadata
- Entry points clearly identified for newcomers

### 2. Better Organization
- Clear hierarchy of documentation
- Consistent categorization across the codebase
- Standard date format for chronological tracking

### 3. Relationship Mapping
- Documents with shared tags/topics can be related
- Technical areas clearly delineated
- Implementation details linked to design documents

### 4. Knowledge Management
- Easier to build documentation indexes
- Documentation gaps become visible
- Historical development path preserved through dates

## Tooling Support

The YAML frontmatter standard enables various tooling possibilities:

1. **Documentation Generators**: Create hierarchical documentation websites
2. **Search Indexes**: Build specialized search functionality
3. **Relationship Maps**: Generate visual document relationship diagrams
4. **Validation Tools**: Ensure conformance to the standard
5. **Tag Clouds**: Visualize the most common topics across the codebase

## Maintenance Guidelines

To maintain the effectiveness of this standard:

1. **Consistency**: Always follow the defined format
2. **Review**: Regularly review and update metadata
3. **Evolution**: Add new tags/keywords as the codebase evolves
4. **Validation**: Periodically check for adherence to the standard
5. **Refinement**: Improve the standard as needs change

## Conclusion

The standardized YAML frontmatter provides a robust foundation for documentation organization and discovery. By consistently applying this standard across all documentation files, we create a more navigable, searchable, and maintainable knowledge base that grows with the project.

This standard should be considered a living document, evolving as the project's documentation needs change over time.
