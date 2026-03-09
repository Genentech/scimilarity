import os, sys
import argparse
import json
import tiledb

import pytorch_lightning as pl
from scimilarity.anndata_data_models import MetricLearningDataModule
from scimilarity.training_models import MetricLearning


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

    datamodule = MetricLearningDataModule(
        train_path=args.train,
        val_path=args.val,
        label_column=args.label_column,
        study_column=args.study_column,
        gene_order_file=args.gene_order,
        batch_size=batch_size,
        num_workers=args.num_workers,
        remove_singleton_classes=True,
        sparse=False,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

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
    parser = argparse.ArgumentParser(description="Train SCimilarity model from anndata")
    parser.add_argument("--train", type=str, help="Training h5ad filename")
    parser.add_argument("--val", type=str, default=None, help="Validation h5ad filename")
    parser.add_argument("--label_column", type=str, default="celltype_name", help="label column")
    parser.add_argument("--study_column", type=str, default="study", help="study column")
    parser.add_argument("--gene_order", type=str, default=None, help="gene order tsv file")
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
