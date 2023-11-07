# CRUST (Conformation Reconstruction Using Spatial Transcriptomics)

The CRUST package provides prediction of chromosome conformation based on spatial transcriptomic.

## Quick Start

### Run CRUST

This example shows the usage of CRUST.

	python REST3d_gem_argv.v0.3.py ./data/SS200000108BR_A3A4_scgem.Spinal_cord_neuron.csv ./results/ Mouse

## Usage

REST3d_gem_argv.v0.3.py [-h] [-p PERCENT] [-l LOOP] [-m MAXDIST] [-c CELLTYPE] [--ctkey CTKEY] [--cikey CIKEY] [--seed SEED] gem_path out_path {Human,Mouse,Axolotls,Monkey}

### positional arguments:

  gem_path              `Path to gem file`

  out_path              `Output path to save results`

  {Human,Mouse,Axolotls,Monkey} `Species of the input data, e.g. Human, Mouse`

### optional arguments:

  -h, --help            `show this help message and exit`

  -p PERCENT, --percent PERCENT  `percent of gene for rotation derivation, default: 0.001`

  -l LOOP, --loop LOOP  `Cluster loops, default: 6`

  -m MAXDIST, --maxdist MAXDIST

                        `max dist of clustering for umap embedding, default: 0.15`

  -c CELLTYPE, --celltype CELLTYPE  `Path of file containing cell types`

  --ctkey CTKEY         `Key of celltype column in the cell type file`

  --cikey CIKEY         `Key of cell id column in the cell type file`

  --seed SEED           `Random seed`