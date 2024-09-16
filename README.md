# Cytocraft

<p align="center">
	<img src=https://github.com/YifeiSheng/Cytocraft/raw/main/figure/Figure1.Overview.png>
</p>

## Overview

The Cytocraft package generates a 3D reconstruction of transcription centers based on subcellular resolution spatial transcriptomics.

## Installaion

```
pip install cytocraft
```

## Interactive Mode Usage

### import
```
import cytocraft.craft as cc
```
### read input 

```
gem_path = './data/mice/example_scgem.csv'
gem = cc.read_gem_as_csv(gem_path, sep=',')
adata = cc.read_gem_as_adata(gem_path, sep=',')
```

### run cytocraft (quick start, see [tutorial](https://github.com/YifeiSheng/Cytocraft/blob/main/tutorial.ipynb) for more details)

```
adata = cc.craft(
  gem=gem,
  adata=adata,
  species='Mice',
  seed=999,
  samplename='example'
  )
```

## CLI Mode Usage
```
python craft.py [-h] [-p/-n PERCENT/NUMBER] [-t GENE_FILTER_THRESH] [-r RMSD_THRESH] [--sep \t] [-c CELLTYPE] [--ctkey CTKEY] [--cikey CIKEY] [--csep \t] [--seed SEED] -i gem_path -o out_path --species {Human,Mouse,Axolotls,Monkey}
```
### Positional arguments:

  -i,--gem_path  `Input: Path of input gene expression matrix file`

  -o,--out_path  `Output: Directory to save results`

  --species  `Species of the input data, e.g. {Human,Mouse,Axolotls,Monkey} `

### Optional arguments:

  -h, --help  `Show this help message and exit`

  -p/-n, --percent/--number  `Percent/Number of anchor gene for rotation derivation, default: 0.001/10`

  -t, --gene_filter_thresh  `The maximum allowable proportion of np.nan values in a column (representing a gene) of the observed transcription centers (Z), default: 0.90`

  -r, --rmsd_thresh  `RMSD threshold. If the computed RMSD value is less than or equal to this threshold, it means the process has reached an acceptable level of similarity or convergence, and the loop is exited. default: 0.01`

  --sep  `Separator of the input gene expression matrix file`

  -c, --celltype  `Path of the annotation file containing cell types, multi-celltype mode only`

  --ctkey  `Key of celltype column in the cell type file, multi-celltype mode only`

  --cikey  `Key of cell id column in the cell type file, multi-celltype mode only`

  --csep  `Separator of the annotation file, multi-celltype mode only, default: \t`

  --seed  `Random seed, default: random int between 0 to 1000`

### One-celltype example:
```
python craft.py -i ./data/SS200000108BR_A3A4_scgem.Spinal_cord_neuron.csv -o ./results/ --species Mouse
```

### Multi-celltype example:
```
python craft.py -i ./data/SSSS200000108BR_A3A4_scgem.csv -o ./results/ --species Mouse --celltype ./data/cell_feature.csv --ctkey cell_type --cikey cell_id
```
