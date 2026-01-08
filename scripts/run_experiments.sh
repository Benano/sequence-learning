#!/usr/bin/env sh

# This script runs a series of experiments with different configurations.
# It scans a specified directory for configuration files
# Then, for each config file it creates a unique output directory.
# Then it copies the config file to the output directory
# Additionally it copies the current script to the output directory for reproducibility.
# Additionally in each directory it creates 2 folders called: figs and artifacts
# Then it runs the script 'run.py' in each output directory with the corresponding config file.

CONFIG_DIR="experiments"
OUTPUT_BASE_DIR="experiments"

# get config file names and put in list
CONFIG_FILES="$CONFIG_DIR"/*.toml
RUN_FILES=*.py
SIM_CONFIG="config.toml"
BASH_FILE="besh_tst.sh"

# Take one input argument which is the name of  a specific experiment to run
# If no argument is given, run all experiments
if [ $# -eq 1 ]; then
    SPECIFIC_EXPERIMENT="$1"
    CONFIG_FILES="$CONFIG_DIR/${SPECIFIC_EXPERIMENT}.toml"
    OUTPUT_BASE_DIR="experiments/${SPECIFIC_EXPERIMENT}"
fi

mkdir -p "$OUTPUT_BASE_DIR"
for CONFIG_FILE in $CONFIG_FILES; do

    # print config file being processed
    echo "Processing config file: $CONFIG_FILE"

    # get basename without the toml extension
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .toml)
    mkdir -p "$OUTPUT_BASE_DIR/${CONFIG_BASENAME}"

    OUTPUT_DIR="$OUTPUT_BASE_DIR/${CONFIG_BASENAME}"

    # copy config file to output directory
    cp "$CONFIG_FILE" "$OUTPUT_DIR/experiment.toml"
    cp "$SIM_CONFIG" "$OUTPUT_DIR/"
    cp "$BASH_FILE" "$OUTPUT_DIR/"
    cp -r patterns "$OUTPUT_DIR/"

    for RUN_FILE in $RUN_FILES; do
        cp "$RUN_FILE" "$OUTPUT_DIR/"
    done

    # Run the bash file
    (cd "$OUTPUT_DIR" && bash "$BASH_FILE") & 
    
done
