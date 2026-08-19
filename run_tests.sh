#!/bin/bash

# ============================================================
# LOCAL LLM TEST RUNNER
# ============================================================

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1


# ============================================================
# VIRTUAL ENVIRONMENT
# ============================================================

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv not found"
    exit 1
fi

source .venv/bin/activate


# ============================================================
# RESULTS DIRECTORY
# ============================================================

mkdir -p results

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

LOG_FILE="results/test_run_${TIMESTAMP}.log"


# ============================================================
# TESTS
# ============================================================

TESTS=(
    "tests/01_test_json.py"
    "tests/02_test_tool_call.py"
    "tests/03_test_tool_execution.py"
    "tests/04_test_multiple_tools.py"
)


# ============================================================
# RUN
# ============================================================

FAILED=0
PASSED=0
TOTAL=0


{

    echo "============================================================"
    echo "LOCAL LLM TEST RUN"
    echo "============================================================"
    echo "Date:    $(date)"
    echo "Project: $PROJECT_DIR"
    echo "Python:  $(python --version 2>&1)"
    echo "============================================================"


    for TEST in "${TESTS[@]}"; do

        TOTAL=$((TOTAL + 1))

        echo
        echo
        echo "############################################################"
        echo "### RUNNING: $TEST"
        echo "############################################################"
        echo

        python "$TEST"

        EXIT_CODE=$?

        echo
        echo "### EXIT CODE: $EXIT_CODE"
        echo "############################################################"

        if [ "$EXIT_CODE" -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi

    done


    # ========================================================
    # SUMMARY
    # ========================================================

    echo
    echo
    echo "============================================================"
    echo "TEST RUN SUMMARY"
    echo "============================================================"

    echo "Total tests:    $TOTAL"
    echo "Passed:         $PASSED"
    echo "Failed:         $FAILED"

    echo

    if [ "$FAILED" -eq 0 ]; then
        echo "OVERALL STATUS: PASS"
    else
        echo "OVERALL STATUS: FAIL"
    fi

    echo
    echo "Log file:       $LOG_FILE"
    echo "Finished:       $(date)"

    echo "============================================================"

} > "$LOG_FILE" 2>&1


# ============================================================
# CONSOLE OUTPUT
# ============================================================

echo
echo "============================================================"
echo "LOCAL LLM TEST RUN FINISHED"
echo "============================================================"
echo
echo "Результат сохранён:"
echo
echo "  $LOG_FILE"
echo


# ============================================================
# FINAL STATUS
# ============================================================

if grep -q "OVERALL STATUS: PASS" "$LOG_FILE"; then
    echo "STATUS: PASS"
    exit 0
else
    echo "STATUS: FAIL"
    exit 1
fi