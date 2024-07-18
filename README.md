# Cytocraft

<p align="center">
	<img src=https://github.com/YifeiSheng/Cytocraft/raw/main/figure/Figure1.Overview.png>
</p>

## Overview

The Cytocraft package generates a 3D reconstruction of transcription centers with subcellular resolution spatial transcriptomics.

## Installaion

```
pip install cytocraft
```

## Interactive Mode Usage

### import
```
from cytocraft.craft import *
```
### read input 

```
gem_path = './data/mice/example_scgem.csv'
gem = read_gem_as_csv(gem_path, sep=',')
adata = read_gem_as_adata(gem_path, sep=',', SN='example')
GeneUIDs = get_GeneUID(gem)
```

### run cytocraft

```
adata = craft(
	gem=gem,
        adata=adata,
        species='Mice',
        seed=999,
        )
```

## CLI Mode Usage
```
python craft.py [-h] [-p PERCENT] [-c CELLTYPE] [--ctkey CTKEY] [--cikey CIKEY] [--seed SEED] gem_path out_path {Human,Mouse,Axolotls,Monkey}
```
### Positional arguments:

  gem_path              `Input: path to gene expression matrix file`

  out_path              `Output: dir path to save results`

  {Human,Mouse,Axolotls,Monkey} `Species of the input data`

### Optional arguments:

  -h, --help     `show this help message and exit`

  -p, --percent  `percent of gene for rotation derivation, default: 0.001`

  -t, --threshold  `The maximum proportion of np.nans allowed in a column(gene) in W, default: 0.90`

  -c, --celltype `Path of file containing cell types, multi-celltype mode only`

  --ctkey `Key of celltype column in the cell type file, multi-celltype mode only`

  --cikey `Key of cell id column in the cell type file, multi-celltype mode only`

  --seed  `Random seed, default: random int between 0 to 1000`

### One-celltype example:
```
python craft.py ./data/SS200000108BR_A3A4_scgem.Spinal_cord_neuron.csv ./results/ Mouse
```

### Multi-celltype example:
```
python craft.py ./data/SSSS200000108BR_A3A4_scgem.csv ./results/ Mouse --celltype ./data/cell_feature.csv --ctkey cell_type --cikey cell_id
```
