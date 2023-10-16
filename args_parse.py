import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="dump/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--load_path", default=None, type=str)

    # dataset
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--config_path", default='configs', type=str)
    parser.add_argument("--dataset", default='GOODHIV', type=str)
    parser.add_argument("--domain", default='scaffold', type=str)
    parser.add_argument("--shift", default='covariate', type=str)

    # VQ
    parser.add_argument("--num_e", default=4000, type=int)
    parser.add_argument("--commitment_weight", default=0.1, type=float)

    # Encoder
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--layer", default=4, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--gnn_type", default='gin', type=str, choices=['gcn', 'gin'])
    parser.add_argument("--pooling_type", default='mean', type=str)

    # Model
    parser.add_argument("--inv_w", default=0.01, type=float)
    parser.add_argument("--reg_w", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)

    # Training
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--epoch", default=200, type=int)

    args = parser.parse_args()

    return args
