#!/bin/bash

# Script to convert eval/outputs/*/*/output.json to CSV and then to HTML
# Skips files that already have corresponding .csv and _viz.html files.

# Paths to scripts
JSON_TO_CSV_SCRIPT="utils/json_to_csv.py"
VISUALIZE_SCRIPT="utils/visualize_results.py"

echo "Starting evaluation output conversion process..."

# Find all output.json files in eval/outputs/
find eval/outputs -name "output.json" | while read -r JSON_FILE; do
    DIR=$(dirname "$JSON_FILE")
    CSV_FILE="$DIR/output.csv"
    HTML_FILE="$DIR/output_viz.html"
    
    echo "--------------------------------------------------"
    echo "Processing directory: $DIR"

    # 1. Convert JSON to CSV if CSV doesn't exist
    if [ ! -f "$CSV_FILE" ]; then
        echo "Converting JSON to CSV: $JSON_FILE"
        uv run "$JSON_TO_CSV_SCRIPT" "$JSON_FILE"
    else
        echo "CSV already exists: $CSV_FILE (Skipping conversion)"
    fi

    # 2. Convert CSV to HTML if HTML doesn't exist (and CSV exists)
    if [ -f "$CSV_FILE" ]; then
        if [ ! -f "$HTML_FILE" ]; then
            echo "Generating HTML visualization: $CSV_FILE"
            uv run "$VISUALIZE_SCRIPT" "$CSV_FILE" --images
        else
            echo "HTML visualization already exists: $HTML_FILE (Skipping visualization)"
        fi
    else
        echo "Error: CSV file $CSV_FILE was not created, skipping visualization."
    fi
done

echo "--------------------------------------------------"
echo "Conversion process complete."
