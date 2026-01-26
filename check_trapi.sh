#!/bin/bash

# Your model list from the colbench config
MODELS=(
    "gpt-5.1-codex_2025-11-13"
    "gpt-5-codex_2025-09-15"
    "gpt-5_2025-08-07"
    "o3-mini_2025-01-31"
    "gpt-5.1_2025-11-13"
    "gpt-5.2_2025-12-11"
    "o4-mini_2025-04-16"
    "gpt-4.1_2025-04-14"
)

ENDPOINT="https://trapi.research.microsoft.com/gcr/shared/openai/v1/responses"

printf "%-35s | %-10s | %-10s | %-10s\n" "MODEL ID" "STATUS" "REM_REQ" "REM_TOK"
echo "----------------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    # Make a tiny request to get headers
    RESPONSE=$(curl -s -i -X POST "$ENDPOINT" \
        -H "Authorization: $AZURE_OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"input\": \"ping\", \"max_output_tokens\": 1}")

    # Extract HTTP status code
    STATUS=$(echo "$RESPONSE" | grep "HTTP/" | awk '{print $2}')
    
    # Extract Remaining Quotas
    REM_REQ=$(echo "$RESPONSE" | grep -i "x-apim-remaining-requests" | awk '{print $2}' | tr -d '\r')
    REM_TOK=$(echo "$RESPONSE" | grep -i "x-apim-remaining-tokens" | awk '{print $2}' | tr -d '\r')

    # Handle empty values
    [ -z "$REM_REQ" ] && REM_REQ="--"
    [ -z "$REM_TOK" ] && REM_TOK="--"

    printf "%-35s | %-10s | %-10s | %-10s\n" "$MODEL" "$STATUS" "$REM_REQ" "$REM_TOK"
done
