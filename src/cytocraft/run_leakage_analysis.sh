#!/bin/bash
#
# Run Cytocraft Segmentation Leakage Robustness Analysis
# This script addresses Reviewer Comment 2 regarding robustness to systematic biological noise
#
# Multi-Cell-Type Tissue Model:
# - Multiple cell types with DIFFERENT chromosome structures (e.g., epithelial, 
#   fibroblast, immune cells each with distinct chromatin organization)
# - Cells of the SAME type have similar structures (with small within-type variation)
# - Cells are positioned in 2D tissue space (optionally clustered by type)
# - Segmentation leakage causes transcript misassignment to spatially neighboring cells
# - Leakage between DIFFERENT cell types introduces very different structural patterns
#
# Usage: ./run_leakage_analysis.sh <genome_structure.tsv> <output_dir>
#

# Default parameters
GENOME_STRUCTURE=${1:-"/data1/shengyifei/ReST3D/results/Simulation/HMEC/chr22.txt"}
OUTPUT_DIR=${2:-"/data1/shengyifei/ReST3D/results/Simulation/HMEC/leakage/"}
N_REPEATS=${3:-10}

# Check if genome structure file is provided
if [ -z "$GENOME_STRUCTURE" ]; then
    echo "Usage: ./run_leakage_analysis.sh <genome_structure.tsv> <output_dir> [n_repeats]"
    echo ""
    echo "Arguments:"
    echo "  genome_structure.tsv  - Path to genome structure file with x, y, z columns"
    echo "  output_dir            - Output directory for results (default: /data1/shengyifei/ReST3D/results/Simulation/HMEC/leakage/)"
    echo "  n_repeats             - Number of repeats per leakage rate (default: 10)"
    echo ""
    echo "This analysis tests Cytocraft robustness to segmentation leakage (transcript"
    echo "misassignment between neighboring cells) at rates from 0% to 30%."
    exit 1
fi

# Check if file exists
if [ ! -f "$GENOME_STRUCTURE" ]; then
    echo "Error: Genome structure file not found: $GENOME_STRUCTURE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "CYTOCRAFT SEGMENTATION LEAKAGE ANALYSIS"
echo "=========================================="
echo "Genome Structure: $GENOME_STRUCTURE"
echo "Output Directory: $OUTPUT_DIR"
echo "Repeats per rate: $N_REPEATS"
echo "Leakage rates: 0%, 5%, 10%, 15%, 20%, 25%, 30%"
echo "=========================================="
echo ""

# Run the leakage analysis
# Multi-cell-type tissue model:
# - 3 cell types with different chromosome structures
# - 10% within-type variation, 50% between-type variation
# - Random spatial mixing of cell types
python "$SCRIPT_DIR/simulation_segmentation_leakage.py" \
    "$GENOME_STRUCTURE" \
    "$OUTPUT_DIR" \
    --leakage_rates 0.0 0.05 0.10 0.15 0.20 0.25 0.30 \
    --n_repeats "$N_REPEATS" \
    --ngene 100 \
    --ncell 300 \
    --rate_cap 0.2 \
    --rate_drop 0.4 \
    --resolution 30 \
    --ngene_rotation 30 \
    --noise 2 \
    --mode random \
    --structure_variation 0.1 \
    --tissue_size 100 \
    --n_cell_types 3 \
    --between_type_variation 0.5 \
    --spatial_clustering 0.0

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "Error: Analysis failed!"
    exit 1
fi
