import os, sys
import argparse
import json
import tiledb

import pytorch_lightning as pl
from scimilarity.tiledb_data_models import CellMultisetDataModule
from scimilarity.training_models import MetricLearning


# Studies to hold out for validation
val_studies = [
    "DS000012424",
    "DS000015749",
    "DS000012615",
    "DS000012500",
    "DS000012585",
    "DS000016072",
    "DS000012588",
    "DS000015711",
    "DS000015743",
    "DS000015752",
    "DS000010060",
    "DS000010475",
    "DS000011735",
    "DS000012592",
    "DS000012595",
]

# Studies to exclude from training
exclude_studies = [
    "DS000010493",
    "DS000011454",
    "DS000015993",
    "DS000011376",
    "DS000015862",
    "DS000012588",
    "DS000016559",
    "DS000012617",
    "DS000016065",
    "DS000010411",
    "DS000012503",
    "DS000011665",
    "DS000010661",
    "DS000017926",
    "DS000010632",
    "DS000017592",
    "DS000015896",
    "DS000010633",
    "DS000016065",
    "DS000016521",
    "DS000013506",
    "DS000017568",
    "DS000018537",
    "DS000016407",
    "DS000012183",
    "DS000014907",
    "DS000015699",
    "DS000010642",
    "DS000016526",
]


def train(args):
    tiledb_base_path = args.tiledb
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    gene_order = args.gene_order
    margin = args.m
    negative_selection = args.t
    triplet_loss_weight = args.w
    lr = args.l
    batch_size = args.b
    n_batches = args.n
    max_epochs = args.e
    cosine_annealing_tmax = args.cosine_annealing_tmax
    suffix = args.suffix

    if cosine_annealing_tmax == 0:
        cosine_annealing_tmax = max_epochs
    
    model_name = (
        f"model_{batch_size}_{margin}_{latent_dim}_{len(hidden_dim)}_{triplet_loss_weight}_{suffix}"
    )
    
    model_folder = args.model_folder
    os.makedirs(model_folder, exist_ok=True)
    log_folder = args.log_folder
    os.makedirs(log_folder, exist_ok=True)
    result_folder = os.path.join(args.result_folder, model_name)
    os.makedirs(result_folder, exist_ok=True)
 
    print(model_name)
    print(args)
    if os.path.isdir(os.path.join(model_folder, model_name)):
        sys.exit(0)

    # Let's filter out cancer datasets based on disease annotation
    sample_tdb = tiledb.open(os.path.join(tiledb_base_path, "sample_metadata"), "r")
    sample_df = sample_tdb.query(attrs=["datasetID", "sampleID", "disease"]).df[:]
    sample_tdb.close()

    cancer_keywords = [
        "cancer",
        "carcinoma",
        "leukemia",
        "myeloma",
        "glioma",
        "tumor",
        "metastati",
        "melanoma",
        "blastoma",
        "sarcoma",
        "cytoma",
        "lymphoma",
        "adenoma",
        "endothelioma",
        "teratoma",
        "lipoma",
        "leiomyoma",
        "meningioma",
        "ependymoma",
    ]
    mask = sample_df["disease"].apply(lambda x: any([k in x for k in cancer_keywords]))
    sample_df = sample_df[mask].set_index(["datasetID", "sampleID"])

    # Exclude specific samples based on the above cancer keyword search
    exclude_samples = {}
    for study, sample in sample_df.index:
        if study not in exclude_samples:
            exclude_samples[study] = []
        exclude_samples[study].append(sample)

    # Set a filter condition for training cells based on columns in the CellArr cell metadata
    filter_condition = f"cellTypeOntologyID!='nan' and total_counts>1000 and n_genes_by_counts>500 and pct_counts_mt<20 and predicted_doublets==0 and cellTypeOntologyID!='CL:0009010'"

    datamodule = CellMultisetDataModule(
        dataset_path=tiledb_base_path,
        gene_order=gene_order,
        val_studies=val_studies,
        exclude_studies=exclude_studies,
        exclude_samples=exclude_samples,
        label_id_column=args.label_id_column,
        study_column=args.study_column,
        sample_column=args.sample_column,
        filter_condition=filter_condition,
        batch_size=batch_size,
        n_batches=n_batches,
        num_workers=args.num_workers,
        sparse=False,
        remove_singleton_classes=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    print(f"Training data size: {datamodule.train_df.shape}")
    print(f"Validation data size: {datamodule.val_df.shape}")

    model = MetricLearning(
        datamodule.n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        margin=margin,
        negative_selection=negative_selection,
        sample_across_studies=(args.cross == 1),
        perturb_labels=(args.perturb == 1),
        perturb_labels_fraction=args.perturb_fraction,
        lr=lr,
        triplet_loss_weight=triplet_loss_weight,
        l1=args.l1,
        l2=args.l2,
        max_epochs=max_epochs,
        cosine_annealing_tmax=cosine_annealing_tmax,
        #track_triplets=result_folder, # uncomment this to track triplet compositions per step
    )

    # Use tensorboard to log training. Modify this based on your preferred logger.
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(
        log_folder,
        name=model_name,
        default_hp_metric=False,
        flush_secs=1,
        version=suffix,
    )

    gpu_idx = args.g

    from pytorch_lightning.callbacks import LearningRateMonitor

    lr_monitor = LearningRateMonitor(logging_interval="step")

    params = {
        "max_epochs": max_epochs,
        "logger": True,
        "logger": logger,
        "accelerator": "gpu",
        "callbacks": [lr_monitor],
        "log_every_n_steps": 1,
        "limit_train_batches": n_batches,
        "limit_val_batches": 10,
        "limit_test_batches": 10,
    }

    trainer = pl.Trainer(**params)

    ckpt_path = os.path.join(log_folder, model_name, suffix, "checkpoints")
    if os.path.isdir(ckpt_path): # resume training if checkpoints exist
        ckpt_files = sorted(
            [x for x in os.listdir(ckpt_path) if x.endswith(".ckpt")],
            key=lambda x: int(x.replace(".ckpt", "").split("=")[-1]),
        )
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=os.path.join(ckpt_path, ckpt_files[-1]),
        )
    else:
        trainer.fit(model, datamodule=datamodule)

    model.save_all(model_path=os.path.join(model_folder, model_name))

    if result_folder is not None:
        test_results = trainer.test(model, datamodule=datamodule)
        if test_results:
            with open(os.path.join(result_folder, f"{model_name}.test.json"), "w+") as fh:
                fh.write(json.dumps(test_results[0]))
    print(model_name)


def main():
    parser = argparse.ArgumentParser(description="Train SCimilarity model")
    parser.add_argument("--tiledb", type=str, help="CellArr tiledb base path")
    parser.add_argument("--label_id_column", type=str, default="cellTypeOntologyID", help="label id column")
    parser.add_argument("--study_column", type=str, default="datasetID", help="study column")
    parser.add_argument("--sample_column", type=str, default="sampleID", help="sample column")
    parser.add_argument("--gene_order", type=str, default="/home/kuot/scratch/scimilarity_gene_order.tsv", help="gene order tsv file")
    parser.add_argument("-g", type=int, default=0, help="gpu index")
    parser.add_argument("-m", type=float, default=0.05, help="triplet loss margin")
    parser.add_argument("-w", type=float, default=0.001, help="triplet loss weight")
    parser.add_argument("-t", type=str, default="semihard", help="negative selection type: [semihard, random, hardest]")
    parser.add_argument("-b", type=int, default=1000, help="batch size, number of cells")
    parser.add_argument("-e", type=int, default=500, help="max epochs")
    parser.add_argument("-n", type=int, default=100, help="number of batches per epoch")
    parser.add_argument("-l", type=float, default=0.005, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=128, help="latent space dim")
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=[1024, 1024, 1024], help="list of hidden layers and sizes")
    parser.add_argument("--input_dropout", type=float, default=0.4, help="input layer dropout p")
    parser.add_argument("--dropout", type=float, default=0.5, help="hidden layer dropout p")
    parser.add_argument("--l1", type=float, default=1e-4, help="l1 regularization lambda")
    parser.add_argument("--l2", type=float, default=0.01, help="l2 regularization lambda")
    parser.add_argument("--cross", type=int, default=1, help="sample across studies, 0: off, 1: on")
    parser.add_argument("--perturb", type=int, default=0, help="perturb labels with parent cell type (if parent exists in training data), 0: off, 1: on")
    parser.add_argument("--perturb_fraction", type=float, default=0.5, help="fraction of labels to attempt to perturb")
    parser.add_argument("--suffix", type=str, default="version_0", help="model name suffix")
    parser.add_argument("--model_folder", type=str, help="where to save model")
    parser.add_argument("--result_folder", type=str, default=None, help="where to save results")
    parser.add_argument("--log_folder", type=str, default="lightning_logs", help="where to save lightning logs")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--cosine_annealing_tmax", type=int, default=0, help="T max for cosine LR annealing, use max epochs if 0")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
