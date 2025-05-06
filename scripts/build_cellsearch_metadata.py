import argparse
import csv
import os, sys
import pandas as pd
import tiledb
from tqdm import tqdm

from scimilarity import CellAnnotation
from scimilarity.ontologies import import_cell_ontology, get_id_mapper

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Build cellsearch metadata")
    parser.add_argument("-t", type=str, help="tiledb base path")
    parser.add_argument("-m", type=str, help="model path")
    parser.add_argument("-b", type=int, default=100000, help="cell buffer size")
    parser.add_argument("--study_column", type=str, default="datasetID", help="study column in tiledb, will be renamed to 'study'")
    parser.add_argument("--sample_column", type=str, default="sampleID", help="sample column in tiledb, will be renamed to 'sample'")
    parser.add_argument("--tissue_column", type=str, default="tissue", help="tissue column in tiledb, will be renamed to 'tissue'")
    parser.add_argument("--disease_column", type=str,default="disease", help="disease column in tiledb, will be renamed to 'disease'")
    parser.add_argument("--knn", type=str, default="labelled_kNN.bin", help="knn filename")
    parser.add_argument("--labels", type=str, default="reference_labels.tsv", help="labels filename")
    parser.add_argument("--safelist_file", type=str,default=None, help="An optional file for a safelist of cell type names in cell type prediction, one per line")
    args = parser.parse_args()
    print(args)

    tiledb_base_path = args.t
    model_path = args.m
    buffer_size = args.b
    study_column = args.study_column
    sample_column = args.sample_column
    tissue_column = args.tissue_column
    disease_column = args.disease_column
 
    # paths
    CELLURI = "cell_metadata"
    cellsearch_path = os.path.join(model_path, "cellsearch")
    os.makedirs(cellsearch_path, exist_ok=True)
 
    # tileDB config
    cfg = tiledb.Config()
    cfg["sm.mem.total_budget"] = 50000000000  # 50G
 
    # cell metadata
    cell_tdb = tiledb.open(os.path.join(tiledb_base_path, CELLURI), "r", config=cfg)
    cell_metadata = cell_tdb.df[:]
    cell_tdb.close()
    cell_metadata = cell_metadata.reset_index(drop=False, names="index")
    cell_metadata = cell_metadata.rename(
        columns={study_column: "study", sample_column: "sample", tissue_column: "tissue", disease_column: "disease"}
    )
    print(f"cellarr metadata: {cell_metadata.shape}")
    print(f"cellarr metadata: {cell_metadata.columns}")

    # map cell type names
    onto = import_cell_ontology()
    id2name = get_id_mapper(onto)
    cell_metadata["author_label"] = cell_metadata["cellTypeOntologyID"].map(id2name).astype(str)

    # training and validation metadata
    train_df = pd.read_csv(os.path.join(model_path, "train_cells.csv.gz"), index_col=0)
    val_df = pd.read_csv(os.path.join(model_path, "val_cells.csv.gz"), index_col=0)
    print(f"training: {train_df.shape}")
    print(f"validation: {val_df.shape}")

    # annotation model
    filenames = {
        "knn": args.knn,
        "celltype_labels": args.labels,
    }
    model = CellAnnotation(model_path=model_path, filenames=filenames)

    if args.safelist_file is not None:
        with open(args.safelist_file, "r") as fh:
            safelist = [line.strip() for line in fh]
        model.safelist_celltypes(safelist)

    embedding_tdb = tiledb.open(os.path.join(cellsearch_path, "cell_embedding"), "r", config=cfg)
    prediction_list = []
    prediction_nn_dist_list = []
    data_type_list = []
    for i in tqdm(range(0, cell_metadata.shape[0], buffer_size)):
        n = min(i + buffer_size, cell_metadata.shape[0])
        df = cell_metadata.iloc[range(i, n)].copy()
        cell_idx = df.index.tolist()
        attr = embedding_tdb.schema.attr(0).name
        embedding = embedding_tdb.query(attrs=[attr], coords=True).multi_index[cell_idx][attr]

        predictions, _, distances, _ = model.get_predictions_knn(embedding, disable_progress=True)
        nn_dist = distances.min(axis=1)
        prediction_list.extend(predictions.values.tolist())
        prediction_nn_dist_list.extend(nn_dist.tolist())
        
        in_train = [x in train_df.index for x in cell_idx]
        in_val = [x in val_df.index for x in cell_idx]
        for j in range(len(in_train)):
            if in_train[j]:
                data_type_list.append("train")
            elif in_val[j]:
                data_type_list.append("test")
            else:
                data_type_list.append("NA")
    embedding_tdb.close()
    
    cell_metadata["data_type"] = data_type_list
    cell_metadata["prediction"] = prediction_list
    cell_metadata["prediction_nn_dist"] = prediction_nn_dist_list

    cell_metadata_tdb_uri = os.path.join(cellsearch_path, "cell_metadata")
    tiledb.from_pandas(cell_metadata_tdb_uri, cell_metadata)

    cell_metadata_tdb = tiledb.open(cell_metadata_tdb_uri, "r", config=cfg)
    print(cell_metadata_tdb.shape)
    print(cell_metadata_tdb.schema)
    cell_metadata_tdb.close()

if __name__ == "__main__":
    main()