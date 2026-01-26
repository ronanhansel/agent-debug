#!/bin/bash

# Ensure we have a token
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "❌ Error: AZURE_OPENAI_API_KEY is not set."
    exit 1
fi

# Format token
[[ $AZURE_OPENAI_API_KEY == Bearer* ]] && AUTH_HEADER="$AZURE_OPENAI_API_KEY" || AUTH_HEADER="Bearer $AZURE_OPENAI_API_KEY"

# We focus on the versions identified in your JSON
MODELS=("gpt-4o_2024-11-20" "gpt-4o_2024-08-06")
ENDPOINTS=("gcr/shared" "msra/shared" "redmond/interactive")

printf "%-20s | %-15s | %-8s | %-10s | %-10s\n" "MODEL VERSION" "REGION" "STATUS" "REM_REQ" "REM_TOK"
echo "---------------------------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    for EP in "${ENDPOINTS[@]}"; do
        BASE_URL="https://trapi.research.microsoft.com/$EP/openai"
        
        # Make request
        RESPONSE=$(curl -s -i -X POST "$BASE_URL/deployments/$MODEL/chat/completions?api-version=2024-10-21" \
            -H "Authorization: $AUTH_HEADER" \
            -H "Content-Type: application/json" \
            -d "{\"messages\": [{\"role\": \"user\", \"content\": \"ping\"}], \"max_tokens\": 1}")

        STATUS=$(echo "$RESPONSE" | grep "HTTP/" | awk '{print $2}')
        
        if [[ "$STATUS" == "404" || "$STATUS" == "400" ]]; then continue; fi

        # Try APIM headers first, then standard Azure RateLimit headers
        REM_REQ=$(echo "$RESPONSE" | grep -iE "x-apim-remaining-requests|x-ratelimit-remaining-requests" | head -n 1 | awk '{print $2}' | tr -d '\r')
        REM_TOK=$(echo "$RESPONSE" | grep -iE "x-apim-remaining-tokens|x-ratelimit-remaining-tokens" | head -n 1 | awk '{print $2}' | tr -d '\r')

        [ -z "$REM_REQ" ] && REM_REQ="--"
        [ -z "$REM_TOK" ] && REM_TOK="--"

        printf "%-20s | %-15s | %-8s | %-10s | %-10s\n" "$MODEL" "$EP" "$STATUS" "$REM_REQ" "$REM_TOK"
    done
done
