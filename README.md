<p align="center">
	<img width="712" height="342" src=https://github.com/YifeiSheng/Cytocraft/blob/main/figure/Figure1.png>
</p>

# Cytocraft

The Cytocraft package provides prediction of chromosome conformation based on spatial transcriptomic.

## Quick Start

### Run Cytocraft

This example shows the usage of Cytocraft.

	python craft.py ./data/SS200000108BR_A3A4_scgem.Spinal_cord_neuron.csv ./results/ Mouse

## Usage

craft.py [-h] [-p PERCENT] [-c CELLTYPE] [--ctkey CTKEY] [--cikey CIKEY] [--seed SEED] gem_path out_path {Human,Mouse,Axolotls,Monkey}

### positional arguments:

  gem_path              `Path to gem file`

  out_path              `Output path to save results`

  {Human,Mouse,Axolotls,Monkey} `Species of the input data`

### optional arguments:

  -h, --help     `show this help message and exit`

  -p, --percent  `percent of gene for rotation derivation, default: 0.001`

  -t, --threshold  `The maximum proportion of np.nans allowed in a column(gene) in W, default: 0.90`

  -c, --celltype `Path of file containing cell types`

  --ctkey `Key of celltype column in the cell type file`

  --cikey `Key of cell id column in the cell type file`

  --seed  `Random seed`
