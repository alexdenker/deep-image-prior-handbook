import traceback
import matplotlib.pyplot as plt
import yaml
import os
import torch
import numpy as np
import random
from PIL import Image
import argparse
from datetime import datetime
import wandb
from dip import (
    DeepImagePrior,
    DeepImagePriorHQS,
    DeepImagePriorTV,
    AutoEncodingSequentialDeepImagePrior,
    SelfGuidanceDeepImagePrior,
    DeepImagePriorHQSDenoiser,
    DeepImagePriorADMMDenoiser,
    DeepImagePriorREDAPG,
    get_unet_model,
    get_walnut_data,
    get_walnut_2d_ray_trafo,
    dict_to_namespace,
    track_best_psnr_output,
    save_images,
    early_stopping,
)
import deepinv as dinv

METHODS_WITH_INNER_STEPS = [
    "tv_hqs",
    "hqs_denoiser",
    "admm_denoiser",
    "apgda_denoiser",
    "redapg",
    "weighted_tv",
    "reddip",
    "aseq"
]
METHODS_WITH_DENOISER = [
    "reddip",
    "apgda_denoiser",
    "redapg",
    "admm_denoiser",
    "hqs_denoiser",
]

use_inp_and_lr = False  # include model_inp and lr in the suffix for the paths and wandb name

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DIP")

    parser.add_argument(
        "--method",
        type=str,
        default="vanilla",
        choices=["vanilla", "tv_hqs", "tv", "aseq", "selfguided", "hqs_denoiser", "admm_denoiser", "redapg"],
        help="DIP method to use",
    )

    parser.add_argument(
        "--model_inp",
        type=str,
        default="fbp",
        choices=["fbp", "random", "adjoint"],
        help="Input to the DIP",
    )

    parser.add_argument("--num_steps", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="adding additional noise to the DIP input",
    )

    parser.add_argument(
        "--exp_weight",
        type=float,
        default=0.0,
        help="Weight for the exponential averaging of the output",
    )

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument(
        "--random_seed_noise",
        type=int,
        default=2,
        help="Random seed for the initial input of the DIP (only used if model_inp = random)",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use WandB logging"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (implies --use_wandb if provided)"
    )

    base_args, remaining = parser.parse_known_args()

    if base_args.wandb_project is not None:
        base_args.use_wandb = True

   
    # Get optional arguments based on the method that are shared among several methods
    if base_args.method in METHODS_WITH_DENOISER:
        parser.add_argument(
            "--denoiser_method",
            type=str,
            default="drunet",
            choices=["tv", "bm3d", "dncnn", "drunet", "gsdrunet"],
            help="Denoiser to use in HQS denoiser DIP",
        )
    if base_args.method in ["admm_denoiser", "apgda_denoiser", "weighted_tv"]:
        parser.add_argument(
            "--admm_weight",
            type=float,
            default=10.0,
            help="ADMM weight for the ADMM denoiser DIP",
        )
    if base_args.method in METHODS_WITH_INNER_STEPS:
        parser.add_argument(
            "--num_inner_steps",
            type=int,
            default=10,
            help="Number of inner optimisation steps",
        )

    if base_args.method in ["selfguided", "aseq", "redapg", "admm_denoiser"]:
        parser.add_argument(
            "--denoise_strength",
            type=float,
            default=0.01,
            help="Denoising strength for the denoising prior",
        )
    if base_args.method in ["tv", "redapg"]:
        if base_args.method == "tv":
            default_reg_strength = 1e-5
        else:  # "redapg"
            default_reg_strength = None  # will be set to denoise_strength / numel in the REDAPG class if not provided
        parser.add_argument(
            "--regularization_strength",
            type=float,
            default=default_reg_strength,
            help="Regularization strength for the regularization prior",
        )
    
    if base_args.method in ["tv_hqs", "hqs_denoiser"]:
        if base_args.method == "tv_hqs":
            default_splitting = 0.5
        else:  # "hqs_denoiser"
            default_splitting = 60.0
        parser.add_argument(
            "--splitting_strength",
            type=float,
            default=default_splitting,
        )
        # Shared default schedule
        parser.add_argument("--reg_min", type=float, default=0.5)
        parser.add_argument("--reg_max", type=float, default=1e-2)

    if base_args.method == "redapg":
        parser.add_argument("--mixing_weight", type=float, default=1.2)

    elif base_args.method == "selfguided":
        parser.add_argument(
            "--num_noise_realisations",
            type=int,
            default=4,
            help="Number of noise realisations for the self-guided DIP",
        )
        parser.add_argument(
            "--lr_z", type=float, default=1e-2, help="Learning rate for the input"
        )
    return parser.parse_args(remaining, namespace=base_args)

def get_reg_strength(args):
    """
    Return the relevant regularisation strength(s) for a given method.
    Returns a dict to support multiple components (eg admm_weight + denoise_strength).
    """
    reg = {}

    # Denoiser-based methods
    if args.method in  ["selfguided", "aseq", "redapg", "admm_denoiser"]:
        reg["denoise_strength"] = args.denoise_strength

    if args.method in ["redapg", "tv"]:
        reg["regularization_strength"] = args.regularization_strength if hasattr(args, "regularization_strength") else None

    # Splitting-based methods
    if args.method in ['tv_hqs', 'hqs_denoiser']:
        reg["splitting_strength"] = args.splitting_strength

    # ADMM-related weights (can co-exist with denoise)
    if args.method in ["admm_denoiser", "apgda_denoiser"]:
        reg["admm_weight"] = args.admm_weight
    return reg


def get_suffix(args, reg_strengths):
    """
    Generate a consistent suffix string that includes relevant method-specific parameters.
    """
    parts = []

    # Denoiser type
    if args.method in METHODS_WITH_DENOISER and args.denoiser_method:
        parts.append(f"{args.denoiser_method}")

    if args.method in METHODS_WITH_INNER_STEPS:
        parts.append(f"inner{args.num_inner_steps}")

    if args.exp_weight > 0:
        parts.append(f"exp{args.exp_weight}")
    else:
        parts.append("noexp")

    if args.method == "selfguided":
        parts.append(f"noise{args.num_noise_realisations}")

    name_map = {
        "denoise_strength": "denoise",
        "tv_strength": "denoise",       
        "splitting_strength": "splitting",
        "admm_weight": "splitting",
    }

    for name, val in reg_strengths.items():
        short_name = name_map.get(name, name)  # fallback to original if not mapped
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        parts.append(f"{short_name}{val_str}")

    return "_" + "_".join(parts)


def make_paths_and_logger(args):
    reg_strength = get_reg_strength(args)
    suffix = get_suffix(args, reg_strength)

    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = f"{args.model_inp}_{args.lr}{suffix}" if use_inp_and_lr else suffix
    # ----- Paths
    base_path = f"results/paper/{args.method}/{suffix}/run_{time_now}"

    paths = {
        "base": base_path,
        "imgs": os.path.join(base_path, "imgs"),
        "logs": os.path.join(base_path, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # ----- Logger config
    from configs.wandb_config import WANDB_ENTITY
    wandb_project = args.wandb_project if args.wandb_project is not None else "DIP_Experiments"
    wandb_entity = WANDB_ENTITY

    wandb_name = f"{args.method}_{suffix}"
    logger_kwargs = {
        "use_wandb": args.use_wandb,
        "project": wandb_project,
        "log_file": os.path.join(paths["logs"], f"log_{time_now}.log"),
        "image_path": paths["imgs"],
        "console_printing": True,
        "image_logging": (
            args.num_inner_steps
            if args.method in METHODS_WITH_INNER_STEPS
            else 25
        ),
        "wandb_config": {
            "project": wandb_project,
            "entity": wandb_entity,
            "name": wandb_name,
            "mode": "online" if args.use_wandb else "disabled",
            "settings": wandb.Settings(code_dir="wandb", _disable_meta=True),
            "dir": "wandb_logs",
            "config": vars(args),
        },
    }
    from dip.logging import FlexibleLogger
    logger = FlexibleLogger(**logger_kwargs)
    return paths, logger


def run_dip(args, logger=None, paths=None):

    device = args.device
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    model_dict = {}
    with open("configs/dip_architecture.yaml", "r") as f:
        model_dict = yaml.safe_load(f)
        model_nsp = dict_to_namespace(model_dict)

    model = get_unet_model(
        in_ch=1,
        out_ch=1,
        scales=model_nsp.scales,
        skip=model_nsp.skip,
        channels=model_nsp.channels,
        use_sigmoid=model_nsp.use_sigmoid,
        use_norm=model_nsp.use_norm,
        activation=model_nsp.activation,
        padding_mode=model_nsp.padding_mode,
        upsample_mode=model_nsp.upsample_mode,
    )
    model.to(device)
    model.train()

    cfg_dict = {}
    with open("configs/walnut_config.yaml", "r") as f:
        data = yaml.safe_load(f)
        cfg_dict["data"] = data

    cfg = dict_to_namespace(cfg_dict)
    cfg.device = device
    settings = {
        "model": model_dict,
        "args": args,
        "date": f"{datetime.now():%Y-%m-%d_%H-%M-%S}",
        "device": device,
        "random_seed": args.random_seed,
        "data_cfg": cfg_dict,
        "args": vars(args),
    }

    ray_trafo = get_walnut_2d_ray_trafo(
        data_path=cfg.data.data_path,
        matrix_path=cfg.data.data_path,
        walnut_id=cfg.data.walnut_id,
        orbit_id=cfg.data.orbit_id,
        angular_sub_sampling=cfg.data.angular_sub_sampling,
        proj_col_sub_sampling=cfg.data.proj_col_sub_sampling,
    )

    ray_trafo.to(device)
    data = get_walnut_data(cfg, ray_trafo=ray_trafo)

    y, x, x_fbp = data[0]

    img = x[0, 0].cpu().numpy() * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(paths["imgs"], "GroundTruth.png"))
    save_dir_img = paths["imgs"]
    if args.model_inp == "fbp":
        z = x_fbp
    elif args.model_inp == "adjoint":
        z = ray_trafo.trafo_adjoint(y).detach()
        z = z / torch.max(z)
    else:
        g = torch.Generator()
        g.manual_seed(args.random_seed_noise)
        z = 0.1 * torch.randn(x.shape, generator=g)

    z = z.to(device)
    
    best_psnr = {'value': 0, 'idx': 0, 'reco': None}
    best_psnr_early_stopping = {'value': 0, 'index': 0, 'reco': None}
    patience = 1000

    if args.method in METHODS_WITH_INNER_STEPS:
        patience = min(100, int(patience / args.num_inner_steps))

    delta = 0.90
    w = 100 
    delta = 0.98 
    variance_list = []

    callbacks = [track_best_psnr_output(best_psnr), 
                 save_images(save_dir_img, skip=10), 
                 early_stopping(patience=patience, delta=delta, w=w, variance_list=variance_list, best_psnr=best_psnr_early_stopping)]
    # Get the denoiser if needed
    if args.method in METHODS_WITH_DENOISER and args.denoiser_method is not None:
        import deepinv as dinv
        if args.denoiser_method == "bm3d":
            denoiser = dinv.models.BM3D()
        elif args.denoiser_method == "drunet":
            denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
        elif args.denoiser_method == "dncnn":
            denoiser = dinv.models.DnCNN(in_channels=1, out_channels=1,  device=device)
        elif args.denoiser_method == "gsdrunet":
            denoiser = dinv.models.GSDRUNet(in_channels=1, out_channels=1, device=device)
        elif args.denoiser_method == "tv":
            reg_denoiser = dinv.optim.prior.TVPrior(n_it_max=100)
            denoiser = lambda x, y: reg_denoiser.prox(x, gamma=y)
        else:
            raise ValueError(f"Denoiser {args.denoiser_method} not recognized. Choose from None, 'tv', 'drunet', 'dncnn', 'bm3d', 'gsdrunet'.")

    if args.method == "vanilla":
        dip = DeepImagePrior(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
    elif args.method == "tv_hqs":
        dip = DeepImagePriorHQS(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            splitting_strength=args.splitting_strength,
            tv_min=args.reg_min,
            tv_max=args.reg_max,
            inner_steps=args.num_inner_steps,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, logger=logger)
    elif args.method == "hqs_denoiser":
        dip = DeepImagePriorHQSDenoiser(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            splitting_strength=args.splitting_strength,
            reg_min=args.reg_min,
            reg_max=args.reg_max,
            num_inner_steps=args.num_inner_steps,
            denoiser=denoiser,
            device=device,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, logger=logger, exp_weight=args.exp_weight)
    elif args.method == "admm_denoiser":
        dip = DeepImagePriorADMMDenoiser(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            denoise_strength=args.denoise_strength,
            num_inner_steps=args.num_inner_steps,
            denoiser=denoiser,
            device=device,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, logger=logger, admm_weight=args.admm_weight, exp_weight=args.exp_weight)
    elif args.method == "apgda_denoiser":
        from dip.model.apgda_denoiser import DeepImagePriorAPGDADenoiser
        dip = DeepImagePriorAPGDADenoiser(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            denoise_strength=args.denoise_strength,
            denoiser=denoiser,
            device=device,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, logger=logger, admm_weight=args.admm_weight)
    elif args.method == "redapg":
        dip = DeepImagePriorREDAPG(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            num_inner_steps=args.num_inner_steps,
            noise_std=args.noise_std,
            denoise_strength=args.denoise_strength,
            regularization_strength=args.regularization_strength,
            denoiser=denoiser,
            device=device,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x, exp_weight=args.exp_weight)
    elif args.method == "tv":
        dip = DeepImagePriorTV(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            tv_strength=args.regularisation_strength,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
    elif args.method == "aseq":
        dip = AutoEncodingSequentialDeepImagePrior(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            denoise_strength=args.denoise_strength,
            callbacks=callbacks,
        )
        x_pred, psnr_list, loss_list = dip.train(
            ray_trafo,
            y,
            z,
            x_gt=x,
            logger=logger,
            num_inner_steps=args.num_inner_steps,
        )
    elif args.method == "selfguided":
        dip = SelfGuidanceDeepImagePrior(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            denoise_strength=args.denoise_strength,
            callbacks=callbacks,
        )
        x_pred, psnr_list, loss_list = dip.train(
            ray_trafo,
            y,
            z,
            x_gt=x,
            logger=logger,
            num_noise_realisations=args.num_noise_realisations,
            exp_weight=args.exp_weight,
            lr_z=args.lr_z,
        )
    else:
        raise NotImplementedError

    # ----- Plotting the reconstructions
    img = x_pred.detach().cpu().numpy()[0, 0] * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(paths["imgs"], "Final_Reconstruction.png"))

    img = best_psnr["reco"][0, 0].numpy() * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(paths["imgs"], "Best_Reconstruction.png"))

    # ----- Saving the results
    results = {}
    results["psnrs"] = [float(x) for x in psnr_list]
    results["loss"] = [float(x) for x in loss_list]
    results["best_psnr"] = float(best_psnr["value"])
    results["best_psnr_idx"] = int(best_psnr["index"])
    results["final_psnr"] = results["psnrs"][-1]
    
    if best_psnr_early_stopping['reco'] is not None:
        img = best_psnr_early_stopping['reco'][0,0].numpy() * 255
        img = img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(paths["imgs"], "Best_Reconstruction_EarlyStopping.png"))
        results["best_psnr_early_stopping"] = float(best_psnr_early_stopping['value'])
        results["best_psnr_early_stopping_idx"] = int(best_psnr_early_stopping['index'])
    
    with open(os.path.join(paths["base"], "results.yaml"), "w") as f:
        yaml.dump(results, f)

    with open(os.path.join(paths["base"], "settings.yaml"), "w") as f:
        yaml.dump(settings, f)

    # ----- Plotting the scalars
    loss_list = np.asarray(loss_list)
    np.save(os.path.join(paths["base"], "loss.npy"), loss_list)

    psnr_list = np.asarray(psnr_list)
    np.save(os.path.join(paths["base"], "psnrs.npy"), psnr_list)
    print(f"Results saved to: {paths['base']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(psnr_list)
    ax1.set(xscale="log", title="PSNR", xlabel="Step", ylabel="PSNR (dB)")
    ax1.grid(True)

    ax2.plot(loss_list)
    ax2.set(
        xscale="log", yscale="log", title="Loss (log)", xlabel="Step", ylabel="Loss"
    )
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(paths["base"], "psnr_and_loss.png"))
    plt.close(fig)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        paths, logger = make_paths_and_logger(args)

        run_dip(args, paths=paths, logger=logger)

    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "error_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/error_log_{timestamp}.txt"
        with open(log_file, "w") as f:
            f.write(f"❌ Uncaught Exception at {timestamp}\n\n")
            traceback.print_exc(file=f)
        print(f"\n❌ Script crashed. Error written to: {log_file}\n")
        traceback.print_exc()
