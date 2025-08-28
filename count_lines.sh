#!/bin/bash

# Script to count lines of code for Python files and lines/words for markdown files
# Author: Generated for cursus project analysis

echo "=========================================="
echo "CURSUS PROJECT LINE COUNT ANALYSIS"
echo "=========================================="
echo

# Function to count lines in a file (excluding empty lines and comments for Python)
count_python_lines() {
    local file="$1"
    if [[ -f "$file" ]]; then
        # Count non-empty lines that are not just comments (lines starting with #)
        grep -v '^\s*#' "$file" | grep -v '^\s*$' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Function to count all lines in a file
count_all_lines() {
    local file="$1"
    if [[ -f "$file" ]]; then
        wc -l < "$file" | tr -d ' '
    else
        echo "0"
    fi
}

# Function to count words in a file
count_words() {
    local file="$1"
    if [[ -f "$file" ]]; then
        wc -w < "$file" | tr -d ' '
    else
        echo "0"
    fi
}

echo "1. PYTHON FILES IN src/cursus PACKAGE"
echo "====================================="

# Find all Python files in src/cursus
python_files_src=($(find src/cursus -name "*.py" -type f | sort))
total_lines_src=0

echo "Individual Python files in src/cursus:"
echo "--------------------------------------"

for file in "${python_files_src[@]}"; do
    lines=$(count_python_lines "$file")
    total_lines_src=$((total_lines_src + lines))
    printf "%-60s %6s lines\n" "$file" "$lines"
done

echo
echo "TOTAL LINES OF CODE in src/cursus package: $total_lines_src"
echo

echo "2. PYTHON TEST FILES IN test FOLDER"
echo "===================================="

# Find all Python files in test folder
python_files_test=($(find test -name "*.py" -type f | sort))
total_lines_test=0

echo "Individual Python test files:"
echo "-----------------------------"

for file in "${python_files_test[@]}"; do
    lines=$(count_python_lines "$file")
    total_lines_test=$((total_lines_test + lines))
    printf "%-60s %6s lines\n" "$file" "$lines"
done

echo
echo "TOTAL LINES OF CODE in test folder: $total_lines_test"
echo

echo "3. MARKDOWN FILES IN slipbox FOLDER"
echo "==================================="

# Find all markdown files in slipbox folder
markdown_files=($(find slipbox -name "*.md" -type f | sort))
total_lines_md=0
total_words_md=0

echo "Individual markdown files:"
echo "--------------------------"

for file in "${markdown_files[@]}"; do
    lines=$(count_all_lines "$file")
    words=$(count_words "$file")
    total_lines_md=$((total_lines_md + lines))
    total_words_md=$((total_words_md + words))
    printf "%-70s %6s lines %8s words\n" "$file" "$lines" "$words"
done

echo
echo "TOTAL LINES in slipbox markdown files: $total_lines_md"
echo "TOTAL WORDS in slipbox markdown files: $total_words_md"
echo

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Python files in src/cursus:     ${#python_files_src[@]} files, $total_lines_src lines of code"
echo "Python files in test:           ${#python_files_test[@]} files, $total_lines_test lines of code"
echo "Markdown files in slipbox:      ${#markdown_files[@]} files, $total_lines_md lines, $total_words_md words"
echo
echo "GRAND TOTAL Python LOC:         $((total_lines_src + total_lines_test)) lines"
echo "=========================================="
