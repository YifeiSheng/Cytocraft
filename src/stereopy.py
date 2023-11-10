import copy
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Optional, Union
from shapely.geometry import Point, MultiPoint
from pathlib import Path
from scipy.sparse import spmatrix, issparse, csr_matrix


class Cell(object):
    def __init__(
        self,
        cell_name: Optional[np.ndarray] = None,
        cell_border: Optional[np.ndarray] = None,
        batch: Optional[Union[np.ndarray, list, int, str]] = None,
    ):
        self._cell_name = cell_name
        self._cell_border = cell_border
        self._batch = self._set_batch(batch) if batch is not None else None
        self.total_counts = None
        self.pct_counts_mt = None
        self.n_genes_by_counts = None

    @property
    def cell_name(self):
        """
        get the name of cell.
        :return: cell name
        """
        return self._cell_name

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.
        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError("cell name must be a np.ndarray object.")
        self._cell_name = name

    @property
    def cell_border(self):
        return self._cell_border

    @cell_border.setter
    def cell_boder(self, cell_border: np.ndarray):
        if not isinstance(cell_border, np.ndarray):
            raise TypeError("cell border must be a np.ndarray object.")
        self._cell_border = cell_border

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch: Union[np.ndarray, list, int]):
        self._batch = self._set_batch(batch)

    def _set_batch(self, batch: Union[np.ndarray, list, int]):
        if batch is None:
            return None

        if (
            not isinstance(batch, np.ndarray)
            and not isinstance(batch, list)
            and not isinstance(batch, int)
            and not isinstance(batch, str)
        ):
            raise TypeError("batch must be np.ndarray or list or int or str")

        if isinstance(batch, int):
            batch = str(batch)
        if isinstance(batch, str):
            return np.repeat(batch, len(self.cell_name))
        else:
            return (np.array(batch) if isinstance(batch, list) else batch).astype("U")

    def sub_set(self, index):
        """
        get the subset of Cell by the index info, the Cell object will be inplaced by the subset.
        :param index: a numpy array of index info.
        :return: the subset of Cell object.
        """
        if self.cell_name is not None:
            self.cell_name = self.cell_name[index]
        if self.cell_boder is not None:
            self.cell_boder = self.cell_boder[index]
        if self.total_counts is not None:
            self.total_counts = self.total_counts[index]
        if self.pct_counts_mt is not None:
            self.pct_counts_mt = self.pct_counts_mt[index]
        if self.n_genes_by_counts is not None:
            self.n_genes_by_counts = self.n_genes_by_counts[index]
        if self.batch is not None:
            self.batch = self.batch[index]
        return self

    def get_property(self, name):
        """
        get the property value by the name.
        :param name: the name of property.
        :return: the property.
        """
        if name == "total_counts":
            return self.total_counts
        if name == "pct_counts_mt":
            return self.pct_counts_mt
        if name == "n_genes_by_counts":
            return self.n_genes_by_counts

    def to_df(self):
        """
        transform Cell object to pd.DataFrame.
        :return: a dataframe of Cell.
        """
        attributes = {
            "total_counts": self.total_counts,
            "pct_counts_mt": self.pct_counts_mt,
            "n_genes_by_counts": self.n_genes_by_counts,
        }
        if self._batch is not None:
            attributes["batch"] = self._batch
        df = pd.DataFrame(attributes, index=self.cell_name)
        return df


class AnnBasedCell(Cell):
    def __init__(
        self,
        based_ann_data: AnnData,
        cell_name: Optional[np.ndarray] = None,
        cell_border: Optional[np.ndarray] = None,
        batch: Optional[Union[np.ndarray, list, int, str]] = None,
    ):
        self.__based_ann_data = based_ann_data
        super(AnnBasedCell, self).__init__(cell_name, cell_border, batch)

    def __str__(self):
        return str(self.__based_ann_data.obs)

    def __repr__(self):
        return self.__str__()

    @property
    def cell_name(self) -> np.ndarray:
        """
        get the name of cell.
        :return: cell name
        """
        return self.__based_ann_data.obs_names.values.astype(str)

    @cell_name.setter
    def cell_name(self, name: np.ndarray):
        """
        set the name of cell.
        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError("cell name must be a np.ndarray object.")
        self.__based_ann_data._inplace_subset_obs(name)

    @property
    def total_counts(self):
        return self.__based_ann_data.obs["total_counts"]

    @total_counts.setter
    def total_counts(self, new_total_counts):
        if new_total_counts is not None:
            self.__based_ann_data.obs["total_counts"] = new_total_counts

    @property
    def pct_counts_mt(self):
        return self.__based_ann_data.obs["pct_counts_mt"]

    @pct_counts_mt.setter
    def pct_counts_mt(self, new_pct_counts_mt):
        if new_pct_counts_mt is not None:
            self.__based_ann_data.obs["pct_counts_mt"] = new_pct_counts_mt

    @property
    def n_genes_by_counts(self):
        return self.__based_ann_data.obs["n_genes_by_counts"]

    @n_genes_by_counts.setter
    def n_genes_by_counts(self, new_n_genes_by_counts):
        if new_n_genes_by_counts is not None:
            self.__based_ann_data.obs["n_genes_by_counts"] = new_n_genes_by_counts


class Gene(object):
    def __init__(self, gene_name: Optional[np.ndarray]):
        self._gene_name = gene_name if gene_name is None else gene_name.astype("U")
        self.n_cells = None
        self.n_counts = None

    @property
    def gene_name(self):
        """
        get the genes name.
        :return: genes name.
        """
        return self._gene_name

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.
        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError("gene name must be a np.ndarray object.")
        self._gene_name = name.astype("U")

    def sub_set(self, index):
        """
        get the subset of Gene by the index info, the Gene object will be inplaced by the subset.
        :param index: a numpy array of index info.
        :return: the subset of Gene object.
        """
        if self.gene_name is not None:
            self.gene_name = self.gene_name[index]
        if self.n_cells is not None:
            self.n_cells = self.n_cells[index]
        if self.n_counts is not None:
            self.n_counts = self.n_counts[index]
        return self

    def to_df(self):
        """
        transform Gene object to pd.DataFrame.
        :return: a dataframe of Gene.
        """
        attributes = {
            "n_counts": self.n_counts,
            "n_cells": self.n_cells,
        }
        df = pd.DataFrame(attributes, index=self.gene_name)
        return df


class AnnBasedGene(Gene):
    def __init__(self, based_ann_data: AnnData, gene_name: Optional[np.ndarray]):
        self.__based_ann_data = based_ann_data
        super(AnnBasedGene, self).__init__(gene_name)

    def __str__(self):
        return str(self.__based_ann_data.var)

    def __repr__(self):
        return self.__str__()

    @property
    def gene_name(self) -> np.ndarray:
        """
        get the genes name.
        :return: genes name.
        """
        return self.__based_ann_data.var_names.values.astype(str)

    @gene_name.setter
    def gene_name(self, name: np.ndarray):
        """
        set the name of gene.
        :param name: a numpy array of names.
        :return:
        """
        if not isinstance(name, np.ndarray):
            raise TypeError("gene name must be a np.ndarray object.")
        self.__based_ann_data._inplace_subset_var(name)


class Data(object):
    def __init__(
        self,
        file_path: Optional[str] = None,
        file_format: Optional[str] = None,
        output: Optional[str] = None,
        partitions: int = 1,
    ):
        self._file = Path(file_path) if file_path is not None else None
        self._partitions = int(partitions)
        self._file_format = file_format
        self.format_range = ["gem", "gef", "mtx", "h5ad", "scanpy_h5ad"]
        self._output = output

    def check(self):
        """
        checking whether the params is in the range.
        :return:
        """
        self.file_check(file=self.file)
        self.format_check(f_format=self.file_format)
        if self.file is not None and self.file_format is None:
            print("the file format must be not None , if the file path is set.")
            raise Exception

    @property
    def output(self):
        """
        the path of output.
        :return:
        """
        return self._output

    @output.setter
    def output(self, path):
        self.output_check(path)
        self._output = path

    @staticmethod
    def file_check(file):
        """
        Check if the file exists.
        :param file: the Path of file.
        :return:
        """
        if file is not None and not file.exists():
            print(f"{str(file)} is not exist, please check!")
            raise FileExistsError

    @property
    def file(self):
        """
        get the file property
        :return:
        """
        return self._file

    @file.setter
    def file(self, path):
        """
        set the file property
        :param path: the file path
        :return:
        """
        if path is None:
            file = path
        elif isinstance(path, str):
            file = Path(path)
        elif isinstance(path, Path):
            file = path
        else:
            raise TypeError
        self.file_check(file=file)
        self._file = file

    @property
    def file_format(self):
        """
        get the file_format property
        :return:
        """
        return self._file_format

    @file_format.setter
    def file_format(self, f_format):
        """
        set the file_format property
        :param f_format: the file format
        :return:
        """
        self.format_check(f_format)
        self._file_format = f_format

    @property
    def partitions(self):
        """
        get the partitions property
        :return:
        """
        return self._partitions

    @partitions.setter
    def partitions(self, partition):
        """
        set the partitions property
        :param partition: the partitions number, which define the cores of multi processes.
        :return:
        """
        self._partitions = partition


class StereoExpData(Data):
    def __init__(
        self,
        file_path: Optional[str] = None,
        file_format: Optional[str] = None,
        bin_type: Optional[str] = None,
        bin_size: int = 100,
        exp_matrix: Optional[Union[np.ndarray, spmatrix]] = None,
        genes: Optional[Union[np.ndarray, Gene]] = None,
        cells: Optional[Union[np.ndarray, Cell]] = None,
        position: Optional[np.ndarray] = None,
        output: Optional[str] = None,
        partitions: int = 1,
        offset_x: Optional[str] = None,
        offset_y: Optional[str] = None,
        attr: Optional[dict] = None,
        merged: bool = False,
    ):
        """
        a Data designed for express matrix of spatial omics. It can directly set the corresponding properties
        information to initialize the data. If the file path is not None, we will read the file information to
        initialize the properties.
        :param file_path: the path of express matrix file.
        :param file_format: the file format of the file_path.
        :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
        :param bin_size: size of bin to merge if bin type is 'bins'.
        :param exp_matrix: the express matrix.
        :param genes: the gene object which contain some info of gene.
        :param cells: the cell object which contain some info of cell.
        :param position: the spatial location.
        :param output: the path of output.
        :param partitions: the number of multi-process cores, used when processing files in parallel.
        :param offset_x: the x of the offset.
        :param offset_y: the y of the offset.
        :param attr: attributions from gef file.
        """
        super(StereoExpData, self).__init__(
            file_path=file_path,
            file_format=file_format,
            partitions=partitions,
            output=output,
        )
        self._exp_matrix = exp_matrix
        self._genes = genes if isinstance(genes, Gene) else Gene(gene_name=genes)
        self._cells = cells if isinstance(cells, Cell) else Cell(cell_name=cells)
        self._position = position
        self._position_offset = None
        self._bin_type = bin_type
        self.bin_size = bin_size
        self._tl = None
        self._plt = None
        self.raw = None
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._attr = attr
        self._merged = merged
        self._sn = self.get_sn_from_path(file_path)

    def get_sn_from_path(self, file_path):
        if file_path is None:
            return None

        from os import path

        return path.basename(file_path).split(".")[0].strip()

    @property
    def tl(self):
        if self._tl is None:
            self._tl = StPipeline(self)
        return self._tl

    def sub_by_index(self, cell_index=None, gene_index=None):
        """
        get sub data by cell index or gene index list.
        :param cell_index: a list of cell index.
        :param gene_index: a list of gene index.
        :return:
        """
        if cell_index is not None:
            self.exp_matrix = self.exp_matrix[cell_index, :]
            self.position = (
                self.position[cell_index, :] if self.position is not None else None
            )
            self.cells = self.cells.sub_set(cell_index)
        if gene_index is not None:
            self.exp_matrix = self.exp_matrix[:, gene_index]
            self.genes = self.genes.sub_set(gene_index)
        return self

    def sub_by_name(
        self,
        cell_name: Optional[Union[np.ndarray, list]] = None,
        gene_name: Optional[Union[np.ndarray, list]] = None,
    ):
        """
        get sub data by cell name list or gene name list.
        :param cell_name: a list of cell name.
        :param gene_name: a list of gene name.
        :return:
        """
        data = copy.deepcopy(self)
        cell_index = (
            [np.argwhere(data.cells.cell_name == i)[0][0] for i in cell_name]
            if cell_name is not None
            else None
        )
        gene_index = (
            [np.argwhere(data.genes.gene_name == i)[0][0] for i in gene_name]
            if gene_name is not None
            else None
        )
        return data.sub_by_index(cell_index, gene_index)

    def check(self):
        """
        checking whether the params is in the range.
        :return:
        """
        super(StereoExpData, self).check()
        self.bin_type_check(self._bin_type)

    @staticmethod
    def bin_type_check(bin_type):
        """
        check whether the bin type is in range.
        :param bin_type: bin type value, 'bins' or 'cell_bins'.
        :return:
        """
        if (bin_type is not None) and (bin_type not in ["bins", "cell_bins"]):
            print(f"the bin type `{bin_type}` is not in the range, please check!")
            raise Exception

    @property
    def shape(self):
        return self.exp_matrix.shape

    @property
    def gene_names(self):
        """
        get the gene names.
        :return:
        """
        return self.genes.gene_name

    @property
    def cell_names(self):
        """
        get the cell names.
        :return:
        """
        return self.cells.cell_name

    @property
    def cell_borders(self):
        return self.cells.cell_boder

    @property
    def genes(self):
        """
        get the value of self._genes.
        :return:
        """
        return self._genes

    @genes.setter
    def genes(self, gene):
        """
        set the value of self._genes.
        :param gene: a object of Gene
        :return:
        """
        self._genes = gene

    @property
    def cells(self):
        """
        get the value of self._cells
        :return:
        """
        return self._cells

    @cells.setter
    def cells(self, cell):
        """
        set the value of self._cells.
        :param cell: a object of Cell
        :return:
        """
        self._cells = cell

    @property
    def exp_matrix(self):
        """
        get the value of self._exp_matrix.
        :return:
        """
        return self._exp_matrix

    @exp_matrix.setter
    def exp_matrix(self, pos_array):
        """
        set the value of self._exp_matrix.
        :param pos_array: np.ndarray or sparse.spmatrix.
        :return:
        """
        self._exp_matrix = pos_array

    @property
    def bin_type(self):
        """
        get the value of self._bin_type.
        :return:
        """
        return self._bin_type

    @bin_type.setter
    def bin_type(self, b_type):
        """
        set the value of self._bin_type.
        :param b_type: the value of bin type, 'bins' or 'cell_bins'.
        :return:
        """
        self.bin_type_check(b_type)
        self._bin_type = b_type

    @property
    def position(self):
        """
        get the value of self._position.
        :return:
        """
        return self._position

    @position.setter
    def position(self, pos):
        """
        set the value of self._position.
        :param pos: the value of position, a np.ndarray .
        :return:
        """
        self._position = pos

    @property
    def position_offset(self):
        return self._position_offset

    @position_offset.setter
    def position_offset(self, position_offset):
        self._position_offset = position_offset

    @property
    def offset_x(self):
        """
        get the x of self._offset_x.
        :return:
        """
        return self._offset_x

    @offset_x.setter
    def offset_x(self, min_x):
        """
        :param min_x: offset of x.
        :return:
        """
        self._offset_x = min_x

    @property
    def offset_y(self):
        """
        get the offset_y of self._offset_y.
        :return:
        """
        return self._offset_y

    @offset_y.setter
    def offset_y(self, min_y):
        """
        :param min_y: offset of y.
        :return:
        """
        self._offset_y = min_y

    @property
    def attr(self):
        """
        get the attr of self._attr.
        :return:
        """
        return self._attr

    @attr.setter
    def attr(self, attr):
        """
        :param attr: dict of attr.
        :return:
        """
        self._attr = attr

    @property
    def merged(self):
        return self._merged

    @merged.setter
    def merged(self, merged):
        self._merged = merged

    @property
    def sn(self):
        return self._sn

    @sn.setter
    def sn(self, sn):
        self._sn = sn

    def to_df(self):
        """
        transform StereoExpData to pd.DataFrame.
        :return:
        """
        df = pd.DataFrame(
            self.exp_matrix.toarray() if issparse(self.exp_matrix) else self.exp_matrix,
            columns=self.gene_names,
            index=self.cell_names,
        )
        return df

    def sparse2array(self):
        """
        transform expression matrix to array if it is parse matrix.
        :return:
        """
        if issparse(self.exp_matrix):
            self.exp_matrix = self.exp_matrix.toarray()
        return self.exp_matrix

    def array2sparse(self):
        """
        transform expression matrix to sparse matrix if it is ndarray
        :return:
        """
        if not issparse(self.exp_matrix):
            self.exp_matrix = csr_matrix(self.exp_matrix)
        return self.exp_matrix

    def __str__(self):
        format_str = f"StereoExpData object with n_cells X n_genes = {self.shape[0]} X {self.shape[1]}"
        format_str += f"\nbin_type: {self.bin_type}"
        if self.bin_type == "bins":
            format_str += f"\n{'bin_size: %d' % self.bin_size}"
        format_str += f"\noffset_x = {self.offset_x}"
        format_str += f"\noffset_y = {self.offset_y}"
        format_cells = []
        for attr_name in [
            ("_cell_name", "cell_name"),
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
        ]:
            if type(attr_name) is tuple:
                real_name, show_name = attr_name[0], attr_name[1]
            else:
                real_name = show_name = attr_name
            # `is not None` is ugly but object in __dict__ may be a pandas.DataFrame
            if self.cells.__dict__.get(real_name, None) is not None:
                format_cells.append(show_name)
        if format_cells:
            format_str += f"\ncells: {format_cells}"
        format_genes = []
        for attr_name in [("_gene_name", "gene_name"), "n_counts", "n_cells"]:
            if type(attr_name) is tuple:
                real_name, show_name = attr_name[0], attr_name[1]
            else:
                real_name = show_name = attr_name
            if self.genes.__dict__.get(real_name, None) is not None:
                format_genes.append(show_name)
        if format_genes:
            format_str += f"\ngenes: {format_genes}"
        # TODO: no decide yet
        # format_str += "\nposition: T"
        format_key_record = {
            key: value for key, value in self.tl.key_record.items() if value
        }
        if format_key_record:
            format_str += f"\nkey_record: {format_key_record}"
        return format_str

    def __repr__(self):
        return self.__str__()

    def issparse(self):
        return issparse(self.exp_matrix)


class StPipeline(object):
    def __init__(self, data):
        """
        A analysis tool sets for StereoExpData. include preprocess, filter, cluster, plot and so on.
        :param data: StereoExpData object.
        """
        self.data = data
        self.result = dict()
        self._raw = None
        self.key_record = {
            "hvg": [],
            "pca": [],
            "neighbors": [],
            "umap": [],
            "cluster": [],
            "marker_genes": [],
        }

    @property
    def raw(self):
        """
        get the StereoExpData whose exp_matrix is raw count.
        :return:
        """
        return self._raw

    @raw.setter
    def raw(self, value):
        """
        set the raw data.
        :param value: StereoExpData.
        :return:
        """
        self._raw = copy.deepcopy(value)

    def reset_raw_data(self):
        """
        reset the self.data to the raw data.
        :return:
        """
        self.data = self.raw

    def raw_checkpoint(self):
        """
        Save the current results to self.raw.
        :param key:
        :param res_key:
        :return:
        """
        self.raw = self.data

    def reset_key_record(self, key, res_key):
        """
        reset key and coordinated res_key in key_record.
        :param key:
        :param res_key:
        :return:
        """
        if key in self.key_record.keys():
            if res_key in self.key_record[key]:
                self.key_record[key].remove(res_key)
            self.key_record[key].append(res_key)
        else:
            self.key_record[key] = [res_key]


def read_gem(file_path, sep="\t", bin_type="cell_bins", bin_size=100, is_sparse=True):
    """
    read the stereo-seq file, and generate the object of StereoExpData.
    :param file_path: input file
    :param sep: separator string
    :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
    :param bin_size: the size of bin to merge. The parameter only takes effect
                    when the value of data.bin_type is 'bins'.
    :param is_sparse: the matrix is sparse matrix if is_sparse is True else np.ndarray
    :return: an object of StereoExpData.
    """
    data = StereoExpData(file_path=file_path, bin_type=bin_type, bin_size=bin_size)
    df = pd.read_csv(
        str(data.file),
        sep=sep,
        comment="#",
        header=0,
        converters={
            "x": int,
            "y": int,
            "MIDCount": float,
            "MIDCounts": float,
            "expr": float,
        },
    )
    # counts
    if "MIDCounts" in df.columns:
        df.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    elif "MIDCount" in df.columns:
        df.rename(columns={"MIDCount": "UMICount"}, inplace=True)
    elif "expr" in df.columns:
        df.rename(columns={"expr": "UMICount"}, inplace=True)
    # cells
    if "CellID" in df.columns:
        df.rename(columns={"CellID": "cell_id"}, inplace=True)
    elif "cell" in df.columns:
        df.rename(columns={"cell": "cell_id"}, inplace=True)
    elif "label" in df.columns:
        df.rename(columns={"label": "cell_id"}, inplace=True)
    # genes
    if "gene" in df.columns:
        df.rename(columns={"gene": "geneID"}, inplace=True)
    df.dropna(inplace=True)
    gdf = None
    gdf = parse_cell_bin_coor(df)
    cells = df["cell_id"].unique()
    genes = df["geneID"].unique()
    cells_dict = dict(zip(cells, range(0, len(cells))))
    genes_dict = dict(zip(genes, range(0, len(genes))))
    rows = df["cell_id"].map(cells_dict)
    cols = df["geneID"].map(genes_dict)
    exp_matrix = csr_matrix(
        (df["UMICount"], (rows, cols)),
        shape=(cells.shape[0], genes.shape[0]),
        dtype=np.float32,
    )
    data.cells = Cell(cell_name=cells)
    data.genes = Gene(gene_name=genes)
    data.exp_matrix = exp_matrix if is_sparse else exp_matrix.toarray()
    data.position = gdf.loc[cells][["x_center", "y_center"]].values
    data.cells.cell_point = gdf.loc[cells]["cell_point"].values
    data.offset_x = df["x"].min()
    data.offset_y = df["y"].min()
    data.attr = {
        "minX": df["x"].min(),
        "minY": df["y"].min(),
        "maxX": df["x"].max(),
        "maxY": df["y"].max(),
        "minExp": data.exp_matrix.toarray().min()
        if is_sparse
        else data.exp_matrix.min(),
        "maxExp": data.exp_matrix.toarray().max()
        if is_sparse
        else data.exp_matrix.min(),
        "resolution": 0,
    }
    return data


def stereo_to_anndata(
    data: StereoExpData, flavor="scanpy", sample_id="sample", reindex=False, output=None
):
    """
    transform the StereoExpData object into Anndata object.
    :param data: StereoExpData object
    :param flavor: 'scanpy' or 'seurat'.
    if you want to convert the output_h5ad into h5seurat for seurat, please set 'seurat'.
    :param sample_id: sample name, which will be set as 'orig.ident' in obs.
    :param reindex: if True, the cell index will be reindex as "{sample_id}:{position_x}_{position_y}" format.
    :param output: path of output_file(.h5ad).
    :return: Anndata object
    """
    from scipy.sparse import issparse

    if data.tl.raw is None:
        print("convert to AnnData should have raw data")
        raise Exception

    exp = (
        data.tl.raw.exp_matrix
        if issparse(data.tl.raw.exp_matrix)
        else csr_matrix(data.tl.raw.exp_matrix)
    )
    cells = data.tl.raw.cells.to_df()
    cells.dropna(axis=1, how="all", inplace=True)
    genes = data.tl.raw.genes.to_df()
    genes.dropna(axis=1, how="all", inplace=True)

    adata = AnnData(
        X=exp,
        dtype=np.float64,
        obs=cells,
        var=genes,
        # uns={'neighbors': {'connectivities_key': 'None','distance_key': 'None'}},
    )
    ##sample id
    print(f"Adding {sample_id} in adata.obs['orig.ident'].")
    adata.obs["orig.ident"] = pd.Categorical(
        [sample_id] * adata.obs.shape[0], categories=[sample_id]
    )
    if data.position is not None:
        print(f"Adding data.position as adata.obsm['spatial'] .")
        adata.obsm["spatial"] = data.position
        # adata.obsm['X_spatial'] = data.position
        print(f"Adding data.position as adata.obs['x'] and adata.obs['y'] .")
        adata.obs["x"] = pd.DataFrame(
            data.position[:, 0], index=data.cell_names.astype("str")
        )
        adata.obs["y"] = pd.DataFrame(
            data.position[:, 1], index=data.cell_names.astype("str")
        )

    if data.sn is not None:
        if isinstance(data.sn, str):
            sn_list = [["-1", data.sn]]
        else:
            sn_list = []
            for bno, sn in data.sn.items():
                sn_list.append([bno, sn])
        adata.uns["sn"] = pd.DataFrame(sn_list, columns=["batch", "sn"])

    for key in data.tl.key_record.keys():
        if len(data.tl.key_record[key]) > 0:
            if key == "hvg":
                res_key = data.tl.key_record[key][-1]
                print(f"Adding data.tl.result['{res_key}'] in adata.var .")
                adata.uns[key] = {"params": {}, "source": "stereopy", "method": key}
                for i in data.tl.result[res_key]:
                    if i == "mean_bin":
                        continue
                    adata.var[i] = data.tl.result[res_key][i]
            elif key == "sct":
                res_key = data.tl.key_record[key][-1]
                # adata.uns[res_key] = {}
                print(f"Adding data.tl.result['{res_key}'] in adata.uns['sct_'] .")
                adata.uns["sct_counts"] = csr_matrix(
                    data.tl.result[res_key][0]["counts"].T
                )
                adata.uns["sct_data"] = csr_matrix(data.tl.result[res_key][0]["data"].T)
                adata.uns["sct_scale"] = csr_matrix(
                    data.tl.result[res_key][0]["scale.data"].T.to_numpy()
                )
                adata.uns["sct_top_features"] = list(
                    data.tl.result[res_key][0]["scale.data"].index
                )
                adata.uns["sct_cellname"] = list(
                    data.tl.result[res_key][1]["umi_cells"].astype("str")
                )
                adata.uns["sct_genename"] = list(
                    data.tl.result[res_key][1]["umi_genes"]
                )
            elif key in ["pca", "umap", "tsne"]:
                # pca :we do not keep variance and PCs(for varm which will be into feature.finding in pca of seurat.)
                res_key = data.tl.key_record[key][-1]
                sc_key = f"X_{key}"
                print(f"Adding data.tl.result['{res_key}'] in adata.obsm['{sc_key}'] .")
                adata.obsm[sc_key] = data.tl.result[res_key].values
            elif key == "neighbors":
                # neighbor :seurat use uns for conversion to @graph slot, but scanpy canceled neighbors of uns at present.
                # so this part could not be converted into seurat straightly.
                for res_key in data.tl.key_record[key]:
                    sc_con = (
                        "connectivities"
                        if res_key == "neighbors"
                        else f"{res_key}_connectivities"
                    )
                    sc_dis = (
                        "distances"
                        if res_key == "neighbors"
                        else f"{res_key}_distances"
                    )
                    print(
                        f"Adding data.tl.result['{res_key}']['connectivities'] in adata.obsp['{sc_con}'] ."
                    )
                    print(
                        f"Adding data.tl.result['{res_key}']['nn_dist'] in adata.obsp['{sc_dis}'] ."
                    )
                    adata.obsp[sc_con] = data.tl.result[res_key]["connectivities"]
                    adata.obsp[sc_dis] = data.tl.result[res_key]["nn_dist"]
                    print(f"Adding info in adata.uns['{res_key}'].")
                    adata.uns[res_key] = {}
                    adata.uns[res_key]["connectivities_key"] = sc_con
                    adata.uns[res_key]["distance_key"] = sc_dis
                    # adata.uns[res_key]['connectivities'] = data.tl.result[res_key]['connectivities']
                    # adata.uns[res_key]['distances'] = data.tl.result[res_key]['nn_dist']
            elif key == "cluster":
                for res_key in data.tl.key_record[key]:
                    print(
                        f"Adding data.tl.result['{res_key}'] in adata.obs['{res_key}'] ."
                    )
                    adata.obs[res_key] = pd.DataFrame(
                        data.tl.result[res_key]["group"].values,
                        index=data.cells.cell_name.astype("str"),
                    )
            elif key == "gene_exp_cluster":
                for res_key in data.tl.key_record[key]:
                    adata.uns[res_key] = data.tl.result[res_key]
            else:
                continue

        if reindex:
            print(f"Reindex adata.X .")
            new_ix = (
                adata.obs["orig.ident"].astype(str)
                + ":"
                + adata.obs["x"].astype(str)
                + "_"
                + adata.obs["y"].astype(str)
            ).tolist()
            adata.obs.index = new_ix
            if "sct_cellname" in adata.uns.keys():
                print(f"Reindex as adata.uns['sct_cellname'] .")
                adata.uns["sct_cellname"] = new_ix

        print(f"Finished conversion to anndata.")

        if output is not None:
            adata.write_h5ad(output)
            print(f"Finished output to {output}")

        return adata


def parse_cell_bin_coor(df):
    gdf = df.groupby("cell_id").apply(lambda x: make_multipoint(x))
    return gdf


def make_multipoint(x):
    p = [Point(i) for i in zip(x["x"], x["y"])]
    mlp = MultiPoint(p).convex_hull
    x_center = mlp.centroid.x
    y_center = mlp.centroid.y
    return pd.Series({"cell_point": mlp, "x_center": x_center, "y_center": y_center})
