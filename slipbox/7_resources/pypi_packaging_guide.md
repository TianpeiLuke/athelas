---
tags:
  - design
  - packaging
  - pypi
  - distribution
  - deployment
  - python
keywords:
  - pyproject.toml
  - twine
  - build system
  - dependency management
  - version management
  - CLI entry points
  - package distribution
  - setuptools
  - wheel
  - source distribution
topics:
  - python packaging
  - pypi distribution
  - package management
  - software deployment
  - dependency resolution
language: python
date of note: 2025-08-27
---

# PyPI Packaging Guide: From Repository to Production Distribution

## Purpose

This document provides a comprehensive guide for packaging Python repositories for distribution on PyPI (Python Package Index). It covers the complete process from analyzing repository structure to successful package upload, with specific focus on modern Python packaging standards using `pyproject.toml`.

## Overview

Modern Python packaging has evolved from `setup.py`-based configurations to declarative `pyproject.toml` files that provide better dependency management, build system specification, and metadata organization. This guide demonstrates the complete packaging workflow implemented for the Athelas repository.

## Packaging Architecture

### Core Components

The modern Python packaging ecosystem consists of several key components:

1. **Build System**: Defines how the package is built (setuptools, wheel)
2. **Package Metadata**: Project information, dependencies, classifiers
3. **File Inclusion**: Specifies which files to include in distributions
4. **Entry Points**: CLI commands and module exports
5. **Version Management**: Centralized version control
6. **Distribution**: Upload and deployment to PyPI

### File Structure

```
repository/
├── pyproject.toml          # Main packaging configuration
├── MANIFEST.in             # File inclusion rules
├── CHANGELOG.md            # Version history
├── README.md               # Package documentation
├── LICENSE                 # License file
├── src/
│   └── package_name/
│       ├── __init__.py     # Package initialization
│       ├── __version__.py  # Version management
│       ├── cli.py          # Command-line interface
│       ├── py.typed        # Type hints marker
│       └── modules/        # Package modules
├── dist/                   # Built distributions (generated)
└── build/                  # Build artifacts (generated)
```

## Implementation Guide

### Step 1: Repository Analysis

Before packaging, analyze the existing repository structure:

#### 1.1 Examine Import Dependencies

```bash
# Search for import statements in source code
find src/ -name "*.py" -exec grep -H "^import\|^from.*import" {} \;
```

**Key Analysis Points**:
- Identify all external dependencies used in the code
- Distinguish between core dependencies and optional features
- Group dependencies by functionality (ML, visualization, cloud, etc.)

#### 1.2 Assess Package Structure

```bash
# List package structure
find src/ -type f -name "*.py" | head -20
```

**Verification Checklist**:
- ✅ Proper `src/` layout with package directory
- ✅ `__init__.py` files in all package directories
- ✅ Logical module organization
- ✅ Clear separation of concerns

### Step 2: Create Packaging Configuration

#### 2.1 Primary Configuration: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "package-name"
version = "0.1.0"
description = "Package description"
authors = [{name = "Author Name", email = "author@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
keywords = ["keyword1", "keyword2", "keyword3"]

# Core dependencies (essential for basic functionality)
dependencies = [
    "dependency1>=1.0.0",
    "dependency2>=2.0.0",
]

# Optional dependencies for specific features
[project.optional-dependencies]
feature1 = [
    "optional-dep1>=1.0.0",
    "optional-dep2>=2.0.0",
]
feature2 = [
    "optional-dep3>=3.0.0",
]
all = [
    "package-name[feature1,feature2]"
]

[project.urls]
Homepage = "https://github.com/username/repository"
Repository = "https://github.com/username/repository"
Documentation = "https://github.com/username/repository/blob/main/README.md"
"Bug Tracker" = "https://github.com/username/repository/issues"

[project.scripts]
package-cli = "package_name.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
package_name = ["py.typed"]
```

#### 2.2 Dependency Strategy

**Core Dependencies**: Essential for basic package functionality
```toml
dependencies = [
    "torch>=2.0.0",           # Based on actual imports
    "numpy>=1.26.0",          # Found in code analysis
    "pandas>=2.1.0",          # Required by processors
]
```

**Optional Dependencies**: Feature-specific extras
```toml
[project.optional-dependencies]
aws = [
    "boto3>=1.39.0",
    "sagemaker>=2.248.0",
]
visualization = [
    "matplotlib>=3.8.0",
    "plotly>=5.0.0",
]
```

### Step 3: Version Management

#### 3.1 Dedicated Version File

Create `src/package_name/__version__.py`:
```python
"""Version information for Package Name."""

__version__ = "0.1.0"
```

#### 3.2 Package Initialization

Update `src/package_name/__init__.py`:
```python
"""
Package Name: Description of the package.
"""

from .__version__ import __version__

__author__ = "Author Name"
__email__ = "author@example.com"
__description__ = "Package description"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
```

### Step 4: File Inclusion Management

#### 4.1 Create `MANIFEST.in`

```
include README.md
include LICENSE
include requirements.txt
include pyproject.toml
recursive-include src/package_name *.py
recursive-include src/package_name *.yaml
recursive-include src/package_name *.yml
recursive-include docs *.md
global-exclude *.pyc
global-exclude __pycache__
global-exclude .DS_Store
global-exclude *.so
global-exclude .git*
```

**File Inclusion Strategy**:
- **Include**: Documentation, configuration, source code
- **Exclude**: Compiled files, cache directories, system files

### Step 5: Command-Line Interface

#### 5.1 CLI Implementation

Create `src/package_name/cli.py`:
```python
"""Command-line interface for Package Name."""
import click
from .__version__ import __version__

@click.group()
@click.version_option(version=__version__)
def main():
    """Package Name: CLI description."""
    pass

@main.command()
def info():
    """Show package information."""
    click.echo(f"Package Name version: {__version__}")
    click.echo("Description of the package")

if __name__ == '__main__':
    main()
```

#### 5.2 Entry Point Configuration

In `pyproject.toml`:
```toml
[project.scripts]
package-cli = "package_name.cli:main"
```

### Step 6: Type Hints Support

#### 6.1 Create Type Marker

Create `src/package_name/py.typed`:
```
# Marker file for PEP 561
# This file indicates that the package supports type hints
```

#### 6.2 Package Data Configuration

In `pyproject.toml`:
```toml
[tool.setuptools.package-data]
package_name = ["py.typed"]
```

### Step 7: Documentation Enhancement

#### 7.1 Comprehensive README.md

```markdown
# Package Name

Brief description of the package.

## Installation

```bash
pip install package-name
```

### Optional Features

```bash
pip install package-name[feature1]    # Specific feature
pip install package-name[all]         # All features
```

## Quick Start

```python
from package_name import main_module

# Usage example
```

## CLI Usage

```bash
package-cli --version
package-cli info
```

## License

MIT License - see [LICENSE](LICENSE) for details.
```

#### 7.2 Version History

Create `CHANGELOG.md`:
```markdown
# Changelog

## [0.1.0] - 2025-08-27

### Added
- Initial release
- Core functionality
- CLI interface
- Documentation

### Features
- Feature 1 description
- Feature 2 description
```

### Step 8: Build and Test Process

#### 8.1 Install Build Tools

```bash
pip install build twine
```

#### 8.2 Build Package

```bash
# Clean previous builds
rm -rf dist/ build/ src/*.egg-info/

# Build distributions
python -m build
```

**Expected Output**:
- `dist/package-name-0.1.0-py3-none-any.whl` (wheel distribution)
- `dist/package-name-0.1.0.tar.gz` (source distribution)

#### 8.3 Local Testing

```bash
# Install locally
pip install dist/package-name-0.1.0-py3-none-any.whl

# Test CLI
package-cli --version
package-cli info

# Test import
python -c "import package_name; print(package_name.__version__)"
```

### Step 9: PyPI Upload

#### 9.1 Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

**Authentication**: Use API token from PyPI account settings

#### 9.2 Verification

After successful upload:
- Package available at: `https://pypi.org/project/package-name/`
- Installation: `pip install package-name`
- CLI available: `package-cli`

## Best Practices

### Dependency Management

1. **Minimal Core Dependencies**: Only include essential dependencies in core
2. **Optional Extras**: Group related optional dependencies
3. **Version Pinning**: Use minimum versions with `>=` for flexibility
4. **Compatibility**: Test across Python versions specified in classifiers

### Version Management

1. **Semantic Versioning**: Follow `MAJOR.MINOR.PATCH` format
2. **Single Source**: Maintain version in dedicated `__version__.py` file
3. **Changelog**: Document all changes in `CHANGELOG.md`
4. **Git Tags**: Tag releases in version control

### File Organization

1. **src/ Layout**: Use `src/` directory for better isolation
2. **Clear Structure**: Organize modules logically
3. **Type Hints**: Include `py.typed` for type checking support
4. **Documentation**: Include comprehensive README and examples

### CLI Design

1. **Consistent Interface**: Use click for professional CLI
2. **Help Text**: Provide clear command descriptions
3. **Version Command**: Always include `--version` option
4. **Error Handling**: Graceful error messages and exit codes

## Troubleshooting

### Common Build Issues

**Issue**: `ModuleNotFoundError` during build
**Solution**: Ensure all `__init__.py` files exist and imports are correct

**Issue**: Missing files in distribution
**Solution**: Update `MANIFEST.in` to include necessary files

**Issue**: Dependency conflicts
**Solution**: Review version constraints and test in clean environment

### Upload Issues

**Issue**: Authentication failure
**Solution**: Use API token instead of username/password

**Issue**: Package name already exists
**Solution**: Choose unique package name or contact PyPI support

**Issue**: File size limits
**Solution**: Exclude unnecessary files, optimize package size

## Security Considerations

1. **API Tokens**: Use scoped API tokens, never commit credentials
2. **Dependencies**: Regularly audit dependencies for vulnerabilities
3. **Code Review**: Review all code before packaging and upload
4. **Signing**: Consider package signing for enhanced security

## Maintenance Workflow

### Regular Updates

1. **Dependency Updates**: Regularly update dependency versions
2. **Security Patches**: Monitor and apply security updates
3. **Python Compatibility**: Test with new Python versions
4. **Documentation**: Keep README and examples current

### Release Process

1. **Version Bump**: Update version in `__version__.py`
2. **Changelog**: Document changes in `CHANGELOG.md`
3. **Testing**: Run full test suite
4. **Build**: Create new distributions
5. **Upload**: Deploy to PyPI
6. **Tag**: Create git tag for release

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Build and Upload to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Upload to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Conclusion

Modern Python packaging with `pyproject.toml` provides a robust, standardized approach to package distribution. This guide demonstrates the complete workflow from repository analysis to PyPI upload, ensuring professional-quality package distribution.

Key success factors:
- Thorough dependency analysis based on actual code
- Modular dependency structure with optional extras
- Professional CLI interface with proper entry points
- Comprehensive documentation and version management
- Proper file inclusion and type hints support

Following this guide ensures packages meet modern Python packaging standards and provide excellent user experience for installation and usage.

## Related Concepts

- **PEP 517/518**: Build system specification standards
- **PEP 621**: Project metadata in pyproject.toml
- **PEP 561**: Type hint distribution
- **Semantic Versioning**: Version numbering scheme
- **Wheel Format**: Binary distribution format

## Cross-References

- [Core Design Principles](core_design_principles.md) - Software engineering principles
- [Unified Zettelkasten Repository Design](unified_zettelkasten_repository_design.md) - Repository architecture
- [Zettelkasten Knowledge Management Principles](zettelkasten_knowledge_management_principles.md) - Knowledge organization
