import argparse
import os

import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy

cfg = tiledb.Config()
cfg["sm.mem.total_budget"] = 50000000000  # 50G

def main():
    parser = argparse.ArgumentParser(description="Build cellsearch knn from embeddings tiledb")
    parser.add_argument("-m", type=str, help="model path")
    parser.add_argument("--cellsearch", type=str, default="cellsearch", help="relative path to the cellsearch folder")
    parser.add_argument("--embeddings", type=str, default="cell_embedding", help="relative path to the cell embeddings folder")
    parser.add_argument("--knn_filename", type=str, default="full_kNN.bin", help="knn filename")
    parser.add_argument("--knn_type", type=str, default="tiledb_vector_search", help="Type of knn: ['hnswlib', 'tiledb_vector_search']")
    parser.add_argument("--ef_construction", type=int, default=1000, help="hnswlib ef construction parameter")
    parser.add_argument("--M_construction", type=int, default=80, help="hnswlib M construction parameter")
    args = parser.parse_args()
    print(args)

    model_path = args.m
    knn_filename = args.knn_filename
    knn_type = args.knn_type
    ef_construction = args.ef_construction
    M = args.M_construction
 
    # embeddings
    cellsearch_path = os.path.join(model_path, args.cellsearch)
    embedding_tdb = tiledb.open(os.path.join(cellsearch_path, args.embeddings), "r", config=cfg)
    attr = embedding_tdb.schema.attr(0).name
    embeddings = embedding_tdb[:][attr]
    embedding_tdb.close()

    # build knn
    knn_fullpath = os.path.join(cellsearch_path, knn_filename)
    if knn_type == "hnswlib":
        # build knn
        n_cells, n_dims = embeddings.shape
        knn = hnswlib.Index(space="cosine", dim=n_dims)
        knn.init_index(max_elements=n_cells, ef_construction=ef_construction, M=M)
        knn.set_ef(ef_construction)
        knn.add_items(embeddings, range(len(embeddings)))
        knn.save_index(os.path.join(cellsearch_path, knn_filename))
    elif knn_type == "tiledb_vector_search":
        knn = vs.ingest(
            index_type="IVF_FLAT",
            index_uri=os.path.join(cellsearch_path, knn_filename),
            input_vectors=embeddings,
            distance_metric=vspy.DistanceMetric.COSINE,
            normalized=True,
            filters=tiledb.FilterList([tiledb.LZ4Filter()])
        )
        knn.vacuum()

        print("Vector array URI:", knn.db_uri, "\n")
        A = tiledb.open(knn.db_uri)
        print("Vector array schema:\n")
        print(A.schema)
        print(A.nonempty_domain())
        A.close()
    
if __name__ == "__main__":
    main()