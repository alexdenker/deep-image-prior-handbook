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
    get_unet_model,
    get_walnut_data,
    get_walnut_2d_ray_trafo,
    dict_to_namespace,
    track_best_psnr_output,
    save_images,
    early_stopping,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DIP")

    parser.add_argument(
        "--method",
        type=str,
        default="vanilla",
        choices=["vanilla", "tv_hqs", "tv", "aseq", "selfguided"],
        help="DIP method to use",
    )

    parser.add_argument(
        "--model_inp",
        type=str,
        default="fbp",
        choices=["fbp", "random", "adjoint"],
        help="Input to the DIP",
    )

    parser.add_argument("--num_steps", type=int, default=20000)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="adding additional noise to the DIP input",
    )

    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument(
        "--random_seed_noise",
        type=int,
        default=2,
        help="Random seed for the initial input of the DIP (only used if model_inp = random)",
    )

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")

    base_args, remaining = parser.parse_known_args()

    if base_args.method == "tv_hqs":
        parser.add_argument("--splitting_strength", type=float, default=60.0)
        parser.add_argument("--tv_min", type=float, default=0.5)
        parser.add_argument("--tv_max", type=float, default=1e-2)
        parser.add_argument("--inner_steps", type=int, default=10)

    elif base_args.method == "tv":
        parser.add_argument("--tv_strength", type=float, default=1e-5)
    elif base_args.method == "aseq":
        parser.add_argument(
            "--denoise_strength",
            type=float,
            default=0.01,
            help="Denoising strength for the denoising prior",
        )
        parser.add_argument(
            "--num_inner_steps",
            type=int,
            default=5,
            help="Number of inner optimisation steps for the aseq DIP",
        )
    elif base_args.method == "selfguided":
        parser.add_argument(
            "--denoise_strength",
            type=float,
            default=0.001,
            help="Denoising strength for the denoising prior",
        )
        parser.add_argument(
            "--num_noise_realisations",
            type=int,
            default=4,
            help="Number of noise realisations for the self-guided DIP",
        )
        parser.add_argument(
            "--exp_weight",
            type=float,
            default=0.99,
            help="Weight for the exponential averaging of the output",
        )
        parser.add_argument(
            "--lr_z", type=float, default=1e-2, help="Learning rate for the input"
        )
    elif base_args.method == "weighted_tv":
        parser.add_argument("--tv_strength", type=float, default=1e-5)
        parser.add_argument("--num_inner_steps", type=int, default=10)
    elif base_args.method == "vanilla":
        pass
    else:
        raise NameError(f"Unknown method: {base_args.method}")
    return parser.parse_args(remaining, namespace=base_args)


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

    noise_std = 0.0

    z = z.to(device)
    y_noise = y.to(device)
    
    best_psnr = {'value': 0, 'idx': 0, 'reco': None}
    best_psnr_early_stopping = {'value': 0, 'index': 0, 'reco': None}
    patience = 1000
    delta = 0.98 
    variance_list = []
    callbacks = [track_best_psnr_output(best_psnr), 
                 save_images(save_dir_img, skip=10), 
                 early_stopping(patience=patience, delta=delta, variance_list=variance_list, best_psnr=best_psnr_early_stopping)]

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
            tv_min=args.tv_min,
            tv_max=args.tv_max,
            inner_steps=args.inner_steps,
            callbacks=callbacks,
        )

        x_pred, psnr_list, loss_list = dip.train(ray_trafo, y, z, x_gt=x)
    elif args.method == "tv":
        dip = DeepImagePriorTV(
            model=model,
            lr=args.lr,
            num_steps=args.num_steps,
            noise_std=args.noise_std,
            tv_strength=args.tv_strength,
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
            num_inner_steps=args.num_inner_steps,
            logger_kwargs=logger_kwargs,
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
    elif args.method == "weighted_tv":
        raise NotImplementedError("Weighted TV DIP not implemented yet")
    else:
        raise NotImplementedError

    img = x_pred.detach().cpu().numpy()[0, 0] * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(paths["imgs"], "Final_Reconstruction.png"))

    img = best_psnr["reco"][0, 0].numpy() * 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(os.path.join(paths["imgs"], "Best_Reconstruction.png"))

    results = {}
    results["psnrs"] = psnr_list
    results["loss"] = loss_list
    results["best_psnr"] = float(best_psnr["value"])
    results["best_psnr_idx"] = int(best_psnr["index"])
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

    loss_list = np.asarray(loss_list)
    np.save(os.path.join(paths["base"], "loss.npy"), loss_list)

    psnr_list = np.asarray(psnr_list)
    np.save(os.path.join(paths["base"], "psnrs.npy"), psnr_list)
    print(f"Results saved to: {paths['base']}")

    steps = np.arange(1, len(psnr_list) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(steps, psnr_list)
    ax1.set(xscale="log", title="PSNR", xlabel="Step", ylabel="PSNR (dB)")
    ax1.grid(True)

    ax2.plot(steps, loss_list)
    ax2.set(
        xscale="log", yscale="log", title="Loss (log)", xlabel="Step", ylabel="Loss"
    )
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(paths["base"], "psnrNloss.png"))
    plt.close(fig)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        wandb_project = f"SelfGuidedDIP"
        wandb_entity = "zkereta"
        base_path = (
            f"results_withskipchannels/{args.method}/{args.model_inp}_{args.denoise_strength}"
        )
        if args.method == "aseq":
            suffix = f"_{args.num_inner_steps}"
        elif args.method == "selfguided":
            suffix = f"_{args.num_noise_realisations}"
        else:
            suffix = ""
        base_path += suffix
        paths = {
            "base": base_path,
            "imgs": os.path.join(base_path, "imgs"),
            "logs": os.path.join(base_path, "logs"),
        }

        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        logger_kwargs = {
            "use_wandb": args.use_wandb,
            "project": wandb_project,
            "log_file": os.path.join(
                paths["logs"], f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
            ),
            "image_path": paths["imgs"],
            "console_printing": True,
            "image_logging": 25,
            "wandb_config": {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": f"{args.method}_{args.model_inp}_{args.denoise_strength}{suffix}",
                "mode": "online" if args.use_wandb else "disabled",
                "settings": wandb.Settings(code_dir="wandb", _disable_meta=True),
                "dir": "wandb_logs",
                "config": vars(args),
            },
        }

        from dip.logging import FlexibleLogger

        logger = FlexibleLogger(**logger_kwargs)
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
