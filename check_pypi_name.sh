#!/usr/bin/env bash

# check_pypi_name.sh
# Usage:
#   ./check_pypi_name.sh name1 [name2 ...]
# Example:
#   ./check_pypi_name.sh autopipe somepkg

PYPI_API="https://pypi.org/pypi"

# Function to check a single name
check_name() {
  local name=$1
  # Query PyPI JSON API; suppress progress, follow redirects, fail silently on HTTP errors
  http_status=$(curl -s -o /dev/null -w "%{http_code}" \
    -A "check-pypi-name-script/1.0 (+https://pypi.org)" \
    "${PYPI_API}/${name}/json")

  if [[ "$http_status" == "200" ]]; then
    echo "✗ '$name' is already taken on PyPI."
  elif [[ "$http_status" == "404" ]]; then
    echo "✓ '$name' is available on PyPI."
  else
    echo "? '$name' returned HTTP status $http_status (could be network issue or rate limiting)."
  fi
}

# Require at least one argument
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 name1 [name2 ...]"
  exit 1
fi

# Loop over provided names
for pkg in "$@"; do
  check_name "$pkg"
done
