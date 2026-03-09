import anndata
import argparse
import hnswlib
import os, sys
import numpy as np
import pandas as pd
import tiledb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Build annotation knn from precomputed embeddings")
    parser.add_argument("-m", type=str, help="model path")
    parser.add_argument("-b", type=int, default=500000, help="cell buffer size")
    parser.add_argument("--annotation", type=str, default="annotation", help="relative path to the annotation folder")
    parser.add_argument("--cellsearch", type=str, default="cellsearch", help="relative path to the cellsearch folder")
    parser.add_argument("--embeddings", type=str, default="cell_embedding", help="relative path to the cell embeddings folder")
    parser.add_argument("--label_column_name", type=str, default="cellTypeName", help="label column name in metadata")
    parser.add_argument("--study_column_name", type=str, default="datasetID", help="study column name in metadata")
    parser.add_argument("--knn", type=str, default="labelled_kNN.bin", help="knn filename")
    parser.add_argument("--labels", type=str, default="reference_labels.tsv", help="labels filename")
    parser.add_argument("--safelist_file", type=str, default=None, help="optional cell type safelist filename")
    parser.add_argument("--ef_construction", type=int, default=1000, help="hnswlib ef construction parameter")
    parser.add_argument("--M_construction", type=int, default=80, help="hnswlib M construction parameter")
    args = parser.parse_args()
    print(args)

    model_path = args.m
    buffer_size = args.b
    label_column_name = args.label_column_name
    study_column_name = args.study_column_name
    knn_filename = args.knn
    label_filename = args.labels
    ef_construction = args.ef_construction
    M = args.M_construction
 
    # tileDB config
    cfg = tiledb.Config()
    cfg["sm.mem.total_budget"] = 50000000000  # 50G

    # training data
    dataframe_path = os.path.join(model_path, "train_cells.csv.gz")
    reference_df = pd.read_csv(dataframe_path, index_col=0)

    if args.safelist_file is not None:
        with open(args.safelist_file, "r") as fh:
            safelist = [line.strip() for line in fh]
        reference_df = reference_df[reference_df[label_column_name].isin(safelist)].copy()
        assert reference_df.shape[0] > 0, "No valid safelist entries in data"

    # precomputed embeddings
    embedding_tdb_uri = os.path.join(model_path, args.cellsearch, args.embeddings)

    embeddings = []
    labels = []
    studies = []
    for i in tqdm(range(0, reference_df.shape[0], buffer_size)):
        j = min(i + buffer_size, reference_df.shape[0])
        df = reference_df.iloc[slice(i, j)].copy()

        embedding_tdb = tiledb.open(embedding_tdb_uri, "r", config=cfg)
        cell_idx = df.index.tolist()
        attr = embedding_tdb.schema.attr(0).name
        embedding = embedding_tdb.query(attrs=[attr], coords=True).multi_index[cell_idx][attr]
        embedding_tdb.close()
        embeddings.append(embedding)
        labels.extend(df[label_column_name].tolist())
        studies.extend(df[study_column_name].tolist())
    embeddings = np.vstack(embeddings) 
    print("embeddings", embeddings.shape)

    annotation_path = os.path.join(model_path, args.annotation)
    os.makedirs(annotation_path, exist_ok=True)

    # save labels
    labels_fullpath = os.path.join(annotation_path, label_filename)
    if os.path.isfile(labels_fullpath):  # backup existing
        os.rename(labels_fullpath, labels_fullpath + ".bak")
    with open(labels_fullpath, "w") as f:
        for i in range(len(labels)):
            f.write(f"{labels[i]}\t{studies[i]}\n")

    # build knn
    n_cells, n_dims = embeddings.shape
    knn = hnswlib.Index(space="cosine", dim=n_dims)
    knn.init_index(max_elements=n_cells, ef_construction=ef_construction, M=M)
    knn.set_ef(ef_construction)
    knn.add_items(embeddings, range(len(embeddings)))

    # save knn
    knn_fullpath = os.path.join(annotation_path, knn_filename)
    if os.path.isfile(knn_fullpath):  # backup existing
        os.rename(knn_fullpath, knn_fullpath + ".bak")
    knn.save_index(knn_fullpath)


if __name__ == "__main__":
    main()