#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Активируем виртуальное окружение
source .venv/bin/activate

# Каталог для результатов
mkdir -p results

# Имя файла с датой и временем
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="results/test_run_${TIMESTAMP}.log"

{
    echo "============================================================"
    echo "LOCAL LLM TEST RUN"
    echo "============================================================"
    echo "Date:    $(date)"
    echo "Project: $PROJECT_DIR"
    echo "Python:  $(python --version 2>&1)"
    echo "============================================================"
    echo

    TESTS=(
        "tests/01_test_json.py"
        "tests/02_test_tool_call.py"
        "tests/03_test_tool_execution.py"
        "tests/04_test_multiple_tools.py"
    )

    FAILED=0

    for TEST in "${TESTS[@]}"; do
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

        if [ $EXIT_CODE -ne 0 ]; then
            FAILED=1
            echo
            echo "!!! TEST FAILED: $TEST !!!"
        fi
    done

    echo
    echo "============================================================"
    echo "TEST RUN FINISHED"
    echo "============================================================"

    if [ $FAILED -eq 0 ]; then
        echo "OVERALL STATUS: PASS"
    else
        echo "OVERALL STATUS: FAIL"
    fi

    echo "Finished: $(date)"
    echo "============================================================"

} > "$LOG_FILE" 2>&1

echo "Тесты завершены."
echo "Результат: $LOG_FILE"

if grep -q "OVERALL STATUS: PASS" "$LOG_FILE"; then
    exit 0
else
    exit 1
fi