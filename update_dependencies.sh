#!/bin/bash

# Update requirements.txt
echo "Checking for new pip dependencies..."
NEW_PIP=$(pip freeze | grep -v -f requirements.txt)

if [[ ! -z "$NEW_PIP" ]]; then
    echo -e "\n# New Dependencies" >> requirements.txt
    echo "$NEW_PIP" >> requirements.txt
    echo "Updated requirements.txt with new dependencies:"
    echo "$NEW_PIP"
else
    echo "No new pip dependencies to add to requirements.txt."
fi

# Update environment.yml
echo "Checking for new micromamba dependencies..."
micromamba list --json | jq -r '.[] | "\(.name)=\(.version)"' > current_env.txt
NEW_ENV=$(comm -13 <(sort environment_cleaned.txt) <(sort current_env.txt))

if [[ ! -z "$NEW_ENV" ]]; then
    echo -e "\n# New Dependencies" >> environment.yml
    while read -r line; do
        PACKAGE=$(echo "$line" | cut -d'=' -f1)
        VERSION=$(echo "$line" | cut -d'=' -f2)
        echo "  - $PACKAGE=$VERSION" >> environment.yml
    done <<< "$NEW_ENV"
    echo "Updated environment.yml with new dependencies:"
    echo "$NEW_ENV"
else
    echo "No new micromamba dependencies to add to environment.yml."
fi

# Clean up temporary file
rm -f current_env.txt
echo "Update process complete!"
