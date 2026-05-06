import argparse
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from load_data import Data
from Experiment import Experiment


def run_experiments(data=None, margin=0.5, noise_reg=0.15, learning_rate=1e-3, dim=32,
                    nneg=10, npos=10, valid_steps=5, num_epochs=500, batch_size=50000, max_norm=1.5, max_grad_norm=5.,
                    optimizer='radam', cuda=True, early_stop=200, real_neg=False, device='cuda:0',
                    step_size=40, gamma=0.9,
                    use_projector=False, proj_dim=None, r_text_embeds=None,
                    use_tangent=True, use_norm=True, use_direction=True,
                    uniform_attn=False, delta_scale=0.1,
                    use_logic_cone=True, cone_scale=0.3,
                    use_consistency_gating=True,
                    use_dynamic_cone=True,
                    use_bc_regularization=True,
                    use_an_modulation=True,
                    use_lc_aggregation=True,
                    static_cone_theta=1.5707963267948966,
                    projector_lr=None,
                    log_path=None,
                    dataset_name=None,
                    checkpoint_path=None,
                    use_multi_hop=True,
                    K_max=3,
                    agg_lambda=0.5,
                    prune_threshold=0.005,
                    grad_ckpt=False,
                    log_path_case=True,
                    path_case_head="Q13909",
                    path_case_relation="P19",
                    path_case_tail="Q65",
                    path_case_topk=20,
                    path_case_every=1,
                    ):
    experiment = Experiment(
        data=data,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dim=dim,
        cuda=cuda,
        nneg=nneg,
        npos=npos,
        max_norm=max_norm,
        optimizer=optimizer,
        valid_steps=valid_steps,
        max_grad_norm=max_grad_norm,
        early_stop=early_stop,
        real_neg=real_neg,
        device=device,
        margin=margin,
        noise_reg=noise_reg,
        step_size=step_size,
        gamma=gamma,
        use_projector=use_projector,
        proj_dim=proj_dim,
        r_text_embeds=r_text_embeds,
        use_tangent=use_tangent,
        use_norm=use_norm,
        use_direction=use_direction,
        uniform_attn=uniform_attn,
        delta_scale=delta_scale,
        use_logic_cone=use_logic_cone,
        cone_scale=cone_scale,
        use_consistency_gating=use_consistency_gating,
        use_dynamic_cone=use_dynamic_cone,
        use_bc_regularization=use_bc_regularization,
        use_an_modulation=use_an_modulation,
        use_lc_aggregation=use_lc_aggregation,
        static_cone_theta=static_cone_theta,
        projector_lr=projector_lr,
        log_path=log_path,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        use_multi_hop=use_multi_hop,
        K_max=K_max,
        agg_lambda=agg_lambda,
        prune_threshold=prune_threshold,
        grad_ckpt=grad_ckpt,
        log_path_case=log_path_case,
        path_case_head=path_case_head,
        path_case_relation=path_case_relation,
        path_case_tail=path_case_tail,
        path_case_topk=path_case_topk,
        path_case_every=path_case_every,
    )
    mrr, hit1, hit3, hit10 = experiment.train_and_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="FB15k-237",
                        help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=700,
                        help="Number of iterations.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=512,
                        help="Batch size.")
    parser.add_argument("--nneg",
                        type=int,
                        default=100,
                        help="Number of negative samples.")
    parser.add_argument("--npos",
                        type=int,
                        default=1,
                        help="Number of positive samples.")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-2,
                        help="Learning rate.")
    parser.add_argument("--dim",
                        type=int,
                        default=256,
                        help="Embedding dimensionality.")
    parser.add_argument('--early_stop', default=100, type=int)
    parser.add_argument('--max_norm', default=5, type=float)
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--max_grad_norm', type=float, default=3)
    parser.add_argument('--real_neg', action='store_true',default = True)
    parser.add_argument('--optimizer',
                        choices=['rsgd', 'radam', 'adam'],
                        default='radam')
    parser.add_argument('--valid_steps', default=50, type=int)
    parser.add_argument("--cuda",
                        type=bool,
                        default=True,
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--device",
                        type=str,
                        default='cuda:1',
                        help="device to use - if cuda = true, (cuda:0, cuda:1, ...), if cuda = false, (cpu)).")
    parser.add_argument("--data",
                        type=str,
                        default='data',
                        help="input data directory")
    parser.add_argument("--noise_reg",
                        type=float,
                        default=1e-2,
                        help="noise level at the regularization of distance")
    parser.add_argument("--step_size",
                        type=int,
                        default=30,
                        help="step size of the scheduler for optimizer")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9,
                        help="gamma of the scheduler for optimizer")
    # ── Semantic Cone Projector ──────────────────────────────────────
    parser.add_argument('--use_projector', action='store_true', default=True,
                        help="Enable the semantic cone projector")
    parser.add_argument('--proj_dim', type=int, default=None,
                        help="Projector output dim (default: dim-1, i.e. spatial dim)")
    parser.add_argument('--no_tangent', action='store_true', default=False,
                        help="Legacy compatibility flag; semantic cone projector ignores view toggles")
    parser.add_argument('--no_norm', action='store_true', default=False,
                        help="Legacy compatibility flag; semantic cone projector ignores view toggles")
    parser.add_argument('--no_direction', action='store_true', default=False,
                        help="Legacy compatibility flag; semantic cone projector ignores view toggles")
    parser.add_argument('--uniform_attn', action='store_true', default=False,
                        help="Legacy compatibility flag; semantic cone projector ignores view attention")
    parser.add_argument('--delta_scale', type=float, default=0.3,
                        help="Scale factor for projector modulation")
    parser.add_argument('--no_logic_cone', action='store_true', default=False,
                        help="Disable semantic logic cone guidance")
    parser.add_argument('--cone_scale', type=float, default=0.3,
                        help="Scale factor for triple-level semantic logic cone bonus")
    parser.add_argument('--no_consistency_gating', action='store_true', default=False,
                        help="Disable exp(-E) consistency gating on projector updates")
    parser.add_argument('--static_cone', '--no_ds_cone', dest='static_cone',
                        action='store_true', default=False,
                        help="Ablation: Static Cone w/o DS-cone (freeze dynamic cone aperture/axis)")
    parser.add_argument('--static_cone_theta', type=float, default=1.5707963267948966,
                        help="Fixed cone half-aperture in radians for --static_cone (default pi/2)")
    parser.add_argument('--no_bc_regularization', '--no_bc_regulation', dest='no_bc_regularization',
                        action='store_true', default=False,
                        help="Ablation: DS-cone w/o Boundary-Constrained Path Regularization")
    parser.add_argument('--no_an_modulation', action='store_true', default=False,
                        help="Ablation: DS-cone w/o Adaptive Neighborhood Modulation")
    parser.add_argument('--no_lc_aggregation', action='store_true', default=False,
                        help="Ablation: DS-cone w/o Lorentz-Constrained Message Aggregation")
    parser.add_argument('--projector_lr', type=float, default=None,
                        help="Separate learning rate for projector (default: same as --lr)")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help="Directory for JSONL training logs (created if missing)")
    parser.add_argument('--exp_name', type=str, default=None,
                        help="Log file basename without .jsonl; default: <dataset>_d<dim>_<timestamp>")
    parser.add_argument('--no_log', action='store_true', default=False,
                        help="Disable writing JSONL log file")
    parser.add_argument('--no_checkpoint', action='store_true', default=False,
                        help="Disable saving best-validation checkpoint (.pt beside the JSONL log)")
    # ── Energy-based Adaptive Multi-hop Reasoning (SIGHT) ────────────
    parser.add_argument('--no_multi_hop', action='store_true', default=False,
                        help="Disable multi-hop reasoning (fall back to single-step refinement)")
    parser.add_argument('--K_max', type=int, default=3,
                        help="Maximum hop depth K_max (default 3)")
    parser.add_argument('--agg_lambda', type=float, default=0.5,
                        help="Residual coefficient lambda for Step-1 neighbour mean aggregation")
    parser.add_argument('--prune_threshold', type=float, default=0.001,
                        help="Count an aggregation neighbour as pruned when its attention weight is below this threshold")
    parser.add_argument('--grad_ckpt', action='store_true', default=False,
                        help="Enable gradient checkpointing inside the K_max loop (trade speed for memory)")
    parser.add_argument('--no_path_case_log', action='store_true', default=False,
                        help="Disable JSONL path_case records for the case-study query")
    parser.add_argument('--path_case_head', type=str, default='Q13909',
                        help="Head entity id for JSONL path_case logging")
    parser.add_argument('--path_case_relation', type=str, default='P19',
                        help="Relation id for JSONL path_case logging")
    parser.add_argument('--path_case_tail', type=str, default='Q65',
                        help="Tail entity id for JSONL path_case logging")
    parser.add_argument('--path_case_topk', type=int, default=20,
                        help="Number of highest-weight paths to keep in each path_case log record")
    parser.add_argument('--path_case_every', type=int, default=1,
                        help="Log one path_case record every N epochs")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = f"{args.data}/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    if args.cuda:
        torch.cuda.set_device(args.device)
    d = Data(data_dir=data_dir)

    r_text_embeds = None
    if args.use_projector:
        r_text_path = os.path.join(data_dir, "relation_text_embeds.npy")
        if os.path.exists(r_text_path):
            r_text_embeds = torch.tensor(np.load(r_text_path), dtype=torch.float32)
            assert r_text_embeds.shape[0] == len(d.relations), \
                f"relation_text_embeds rows ({r_text_embeds.shape[0]}) != num relations ({len(d.relations)})"
            print(f"Loaded relation text embeddings: {r_text_path}  shape={r_text_embeds.shape}")
        else:
            print("No relation_text_embeds.npy found; using learnable relation embeddings for projector.")

    print(args)

    # ── SIGHT Multi-hop banner ───────────────────────────────────────
    _multi_hop_on = not args.no_multi_hop
    print("=" * 72)
    print("  SIGHT Energy-based Adaptive Multi-hop Reasoning")
    print("  multi_hop    : %s" % ("ENABLED" if _multi_hop_on else "disabled"))
    print("  K_max        : %d" % args.K_max)
    print("  agg_lambda λ : %.3f" % args.agg_lambda)
    print("  prune_thres  : %.6g" % args.prune_threshold)
    print("  grad_ckpt    : %s" % ("on" if args.grad_ckpt else "off"))
    print("  use_projector: %s  |  use_logic_cone: %s" %
          (args.use_projector, not args.no_logic_cone))
    print("  ablations    : DS=%s | BC=%s | AN=%s | LC=%s" % (
          "static" if args.static_cone else "dynamic",
          "off" if args.no_bc_regularization else "on",
          "off" if args.no_an_modulation else "on",
          "off" if args.no_lc_aggregation else "on"))
    print("=" * 72)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = args.exp_name or f"{dataset}_d{args.dim}_{stamp}"
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]

    log_path = None
    if not args.no_log:
        log_path = os.path.join(args.log_dir, stem + ".jsonl")

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = os.path.join(args.log_dir, stem + ".pt")

    run_experiments(data=d, margin=args.margin, noise_reg=args.noise_reg,
                    learning_rate=args.lr, dim=args.dim, nneg=args.nneg, npos=args.npos,
                    valid_steps=args.valid_steps, num_epochs=args.num_epochs, batch_size=args.batch_size,
                    max_norm=args.max_norm, max_grad_norm=args.max_grad_norm, optimizer=args.optimizer,
                    early_stop=args.early_stop, real_neg=args.real_neg, device=args.device,
                    step_size=args.step_size, gamma=args.gamma,
                    use_projector=args.use_projector, proj_dim=args.proj_dim,
                    r_text_embeds=r_text_embeds,
                    use_tangent=not args.no_tangent, use_norm=not args.no_norm,
                    use_direction=not args.no_direction,
                    uniform_attn=args.uniform_attn, delta_scale=args.delta_scale,
                    use_logic_cone=not args.no_logic_cone,
                    cone_scale=args.cone_scale,
                    use_consistency_gating=not args.no_consistency_gating,
                    use_dynamic_cone=not args.static_cone,
                    use_bc_regularization=not args.no_bc_regularization,
                    use_an_modulation=not args.no_an_modulation,
                    use_lc_aggregation=not args.no_lc_aggregation,
                    static_cone_theta=args.static_cone_theta,
                    projector_lr=args.projector_lr,
                    log_path=log_path,
                    dataset_name=dataset,
                    checkpoint_path=checkpoint_path,
                    use_multi_hop=not args.no_multi_hop,
                    K_max=args.K_max,
                    agg_lambda=args.agg_lambda,
                    prune_threshold=args.prune_threshold,
                    grad_ckpt=args.grad_ckpt,
                    log_path_case=not args.no_path_case_log,
                    path_case_head=args.path_case_head,
                    path_case_relation=args.path_case_relation,
                    path_case_tail=args.path_case_tail,
                    path_case_topk=args.path_case_topk,
                    path_case_every=args.path_case_every)
