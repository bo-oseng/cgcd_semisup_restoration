import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official implementation of `Proxy Anchor Loss for Deep Metric Learning`"
        + " Our code is modified from `https://github.com/dichotomies/proxy-nca`"
    )
    parser.add_argument("--LOG_DIR", default="./logs", help="Path to log folder")
    parser.add_argument("--dataset", default="deg", help="Training dataset, e.g. cub, cars, SOP, Inshop")
    parser.add_argument(
        "--embedding_size",
        default=1024,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch_size",
        default=120,
        type=int,
        dest="sz_batch",
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        dest="nb_epochs",
        help="Number of training epochs.",
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="ID of GPU that is used for training.")
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument("--model", default="dinov3", help="Model for training")
    parser.add_argument("--loss", default="Proxy_Anchor", help="Criterion for training")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay setting")
    parser.add_argument("--lr_decay_step", default=5, type=int, help="Learning decay step setting")
    parser.add_argument(
        "--lr_decay_gamma",
        default=0.5,
        type=float,
        help="Learning decay gamma setting",
    )
    parser.add_argument("--alpha", default=32, type=float, help="Scaling Parameter setting")
    parser.add_argument("--mrg", default=0.1, type=float, help="Margin parameter setting")
    parser.add_argument("--warm", default=5, type=int, help="Warmup training epochs")
    parser.add_argument("--bn_freeze", default=True, type=bool, help="Batch normalization parameter freeze")
    parser.add_argument("--l2_norm", default=True, type=bool, help="L2 normalization flag")
    parser.add_argument("--remark", default="", help="Any remark")
    parser.add_argument("--use_split_modlue", type=bool, default=True)
    parser.add_argument("--use_GM_clustering", type=bool, default=True)
    parser.add_argument("--exp", type=str, default="debug")
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="degradations_with_real",
        help="Target dataset override",
    )
    parser.add_argument("--thres", type=float, default=0.0, help="Old/new split cosine similarity threshold")
    parser.add_argument(
        "--preference",
        type=float,
        default=None,
        help="AffinityPropagation preference",
    )
    parser.add_argument("--cgcd_ckpt", default=None, help="CGCD checkpoint path")
    parser.add_argument(
        "--confidence_thres",
        type=float,
        default=0.03,
        help="Initial split confidence threshold",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
