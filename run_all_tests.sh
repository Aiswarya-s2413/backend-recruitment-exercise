#!/bin/bash

# This script automates running tests for all services in the project.
# It navigates into each service directory, installs dependencies using Poetry,
# and then executes the corresponding test suite with pytest.

set -e # Exit immediately if a command exits with a non-zero status.

# Get the absolute path of the script's directory to resolve paths correctly
BASE_DIR=$(cd "$(dirname "$0")" && pwd)

# Define services and their corresponding test files
declare -A services
services=(
    ["pdf_service"]="tests/test_pdf_service.py"
    ["rag_module"]="tests/test_rag_module.py"
    ["aws_service"]="tests/test_aws_service.py"
    ["metrics_lambda"]="tests/test_metrics_lambda.py"
)

echo "========================================="
echo "      RUNNING ALL PROJECT TESTS"
echo "========================================="
echo

# Loop through each service, install dependencies, and run tests
for service in "${!services[@]}"; do
    test_file="${services[$service]}"
    echo "-----------------------------------------"
    echo "Testing service: $service"
    echo "-----------------------------------------"
    
    (cd "$BASE_DIR/$service" && echo "Installing dependencies for $service..." && poetry install --no-root && echo "Running tests..." && poetry run pytest "$BASE_DIR/$test_file" -v)
    
    echo "✅ Tests for $service completed successfully."
    echo
done

echo "========================================="
echo "  ALL TESTS PASSED SUCCESSFULLY!"
echo "========================================="