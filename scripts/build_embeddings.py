import argparse
import os
import shutil
import sys
from tqdm import tqdm

import tiledb
import numpy as np
from scipy.sparse import coo_matrix, diags

from scimilarity import CellEmbedding
from scimilarity.utils import write_array_to_tiledb, optimize_tiledb_array

cfg = tiledb.Config()
cfg["sm.mem.total_budget"] = 50000000000  # 50G

def get_expression(matrix_tdb, matrix_shape, cell_idx, gene_indices, target_sum=1e4):
    results = matrix_tdb[cell_idx, :]
    counts = coo_matrix((results["data"], (results["cell_index"], results["gene_index"])), shape=matrix_shape).tocsr()
    counts = counts[cell_idx, :]
    counts = counts[:, gene_indices]

    X = counts.astype(np.float32)

    # normalize to target sum
    row_sums = np.ravel(X.sum(axis=1))  # row sums as a 1D array
    # avoid division by zero by setting zero sums to one (they will remain zero after normalization)
    row_sums[row_sums == 0] = 1
    # create a sparse diagonal matrix with the inverse of the row sums
    inv_row_sums = diags(1 / row_sums).tocsr()
    # normalize the rows to sum to 1
    normalized_matrix = inv_row_sums.dot(X)
    # scale the rows sum to target_sum
    X = normalized_matrix.multiply(target_sum)
    X = X.log1p()

    return X

def main():
    parser = argparse.ArgumentParser(description="Build embeddings tiledb")
    parser.add_argument("-t", type=str, help="CellArr base path")
    parser.add_argument("-m", type=str, help="model path")
    parser.add_argument("-b", type=int, default=100000, help="batch size")
    args = parser.parse_args()
    print(args)

    model_path = args.m
    batch_size = args.b
 
    # model
    ce = CellEmbedding(model_path)
    cellsearch_path = os.path.join(model_path, "cellsearch")
    os.makedirs(cellsearch_path, exist_ok=True)
    embedding_tdb_uri = os.path.join(cellsearch_path, "cell_embedding")

    # cellarr
    tiledb_base_path = args.t
    GENEURI = "gene_annotation"
    COUNTSURI = "counts"

    # gene space alignment
    gene_tdb = tiledb.open(os.path.join(tiledb_base_path, GENEURI), "r", config=cfg)
    genes = gene_tdb.query(attrs=["cellarr_gene_index"]).df[:]["cellarr_gene_index"].tolist()
    gene_tdb.close()
    gene_indices = [genes.index(x) for x in ce.gene_order]

    # counts matrix
    matrix_tdb_uri = os.path.join(tiledb_base_path, COUNTSURI)
    matrix_tdb = tiledb.open(os.path.join(tiledb_base_path, COUNTSURI), "r", config=cfg)
    matrix_shape = (matrix_tdb.nonempty_domain()[0][1] + 1, matrix_tdb.nonempty_domain()[1][1] + 1)
    print("Cell counts:", matrix_shape)

    # array schema
    xdimtype = np.uint32
    ydimtype = np.uint32
    value_type = np.float32
    
    xdim = tiledb.Dim(name="x", domain=(0, matrix_shape[0] - 1), tile=10000, dtype=xdimtype)
    ydim = tiledb.Dim(name="y", domain=(0, ce.latent_dim - 1), tile=ce.latent_dim, dtype=ydimtype)
    dom = tiledb.Domain(xdim, ydim)
    
    attr = tiledb.Attr(
        name="data",
        dtype=value_type,
        filters=tiledb.FilterList([tiledb.LZ4Filter()]),
    )
    
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        cell_order="row-major",
        tile_order="row-major",
        attrs=[attr],
    )
    
    if os.path.exists(embedding_tdb_uri):
        shutil.rmtree(embedding_tdb_uri)
    tiledb.Array.create(embedding_tdb_uri, schema)

    # write to array
    embeddings = []
    embedding_tdb = tiledb.open(embedding_tdb_uri, "w", config=cfg)
    for i in tqdm(range(0, matrix_shape[0], batch_size)):
        j = min(i + batch_size, matrix_shape[0])
        cell_idx = slice(i, j)
        X = get_expression(matrix_tdb, matrix_shape, cell_idx, gene_indices)
        embedding = ce.get_embeddings(X).astype(value_type)
        embedding_tdb[cell_idx] = embedding
        embeddings.append(embedding)
    matrix_tdb.close()
    embedding_tdb.close()
 
    embedding_tdb = tiledb.open(embedding_tdb_uri, "r", config=cfg)
    print("Embeddings tiledb:", embedding_tdb.nonempty_domain()) 
    embeddings = np.vstack(embeddings)
    print("Embeddings numpy:", embeddings.shape)

    optimize_tiledb_array(embedding_tdb_uri)

if __name__ == "__main__":
    main()