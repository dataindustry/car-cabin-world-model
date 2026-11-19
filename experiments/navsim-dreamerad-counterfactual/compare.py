"""Run two action-conditioned DreamerAD rollouts on the same NAVSIM history."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def trajectory(start_speed, acceleration, time_s):
    moving_time = time_s if acceleration == 0 else np.minimum(
        time_s, start_speed / -acceleration
    )
    distance = start_speed * moving_time + 0.5 * acceleration * moving_time**2
    speed = np.maximum(0, start_speed + acceleration * time_s)
    poses = np.repeat(np.eye(4)[None], len(time_s), axis=0)
    poses[:, 0, 3] = distance
    return poses, distance, speed


def load_observations(scene_path, scene, image_size, device):
    height, width = image_size
    images = []
    for filename, expected_digest in zip(scene["history_files"], scene["history_sha256"]):
        path = scene_path.parent / filename
        if sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"History image checksum mismatch: {path}")
        with Image.open(path) as original:
            image = original.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        images.append(np.array(image))
    rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    return (rgb.to(device, dtype=torch.float32) / 255 * 2 - 1).unsqueeze(0)


def comparison_panels(predictions, cases, times):
    labels = {"brake": "Brake", "hold": "Hold speed"}
    font = ImageFont.load_default(size=20)
    for i, seconds in enumerate(times):
        sheet = Image.new("RGB", (1024, 296), "#171d26")
        draw = ImageDraw.Draw(sheet)
        for col, case in enumerate(labels):
            preview = Image.fromarray(predictions[case][i]).resize(
                (512, 256), Image.Resampling.LANCZOS
            )
            x = 512 * col
            sheet.paste(preview, (x, 40))
            distance = cases[case]["position_m"][i]
            speed = cases[case]["speed_mps"][i]
            draw.text(
                (x + 12, 8),
                f"{labels[case]}  |  +{seconds:.1f}s  |  {distance:.1f} m  |  {speed:.1f} m/s",
                fill="white",
                font=font,
            )
        yield sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deceleration", type=float, default=2.0)
    parser.add_argument("--dit-steps", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=4)
    args = parser.parse_args()
    if not np.isfinite(args.deceleration) or args.deceleration <= 0:
        parser.error("--deceleration must be positive and finite")

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.environ["OPENSCENE_DATA_ROOT"] = str(args.scene.resolve().parent)
    os.environ["DREAMERAD_ROOT"] = str(source)
    os.environ["DREAMERAD_VAE_CKPT"] = str(args.vae.resolve())

    from models.modules.tokenizer import VAETokenizer
    from models.world_model import DreamerADWorldModel, load_world_model_checkpoint
    from scripts.visualize import generate_future_frames, seed_everything
    from utils.config_utils import Config

    config = Config.fromfile(str(source / "configs/inference.py"))
    config.dit_steps = args.dit_steps
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    history = np.asarray(scene["history_transforms_local"])
    if history.shape != (config.condition_frames, 4, 4):
        raise ValueError("Unexpected number of observation transforms")
    if len(scene["history_files"]) != config.condition_frames:
        raise ValueError("Unexpected number of observation images")
    if not np.allclose(history[-1], np.eye(4), atol=1e-6):
        raise ValueError("The last observed transform must anchor the local frame")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    images = load_observations(args.scene, scene, config.image_size, device)

    print("Loading DreamerAD model", flush=True)
    model = DreamerADWorldModel(config, 0, config.condition_frames).bfloat16()
    checkpoint_report = load_world_model_checkpoint(model, args.checkpoint)
    model.eval().requires_grad_(False)
    print(f"Loaded DreamerAD weights: {checkpoint_report}", flush=True)
    tokenizer = VAETokenizer(config, 0)
    tokenizer.eval().requires_grad_(False)

    fps = config.downsample_fps
    times = np.arange(1, config.traj_len + 1) / fps
    cases = {}
    predictions = {}
    end_rng = {}
    for case, acceleration in (("hold", 0.0), ("brake", -args.deceleration)):
        future, distance, speed = trajectory(scene["initial_speed_mps"], acceleration, times)
        transforms = torch.tensor(
            np.concatenate([history, future]), device=device, dtype=torch.float32
        ).unsqueeze(0)
        seed_everything(args.seed)
        print(f"Rolling out {case}", flush=True)
        started = time.perf_counter()
        predictions[case] = generate_future_frames(model, tokenizer, images, transforms, config)
        torch.cuda.synchronize()
        end_rng[case] = torch.cuda.get_rng_state(device)
        if len(predictions[case]) != config.traj_len:
            raise ValueError(f"Unexpected rollout length for {case}")
        cases[case] = {
            "acceleration_command_mps2": acceleration,
            "position_m": distance.tolist(),
            "speed_mps": speed.tolist(),
            "elapsed_s": round(time.perf_counter() - started, 3),
        }

    if not torch.equal(end_rng["brake"], end_rng["hold"]):
        raise RuntimeError("The two rollouts consumed different CUDA random sequences")
    difference = [
        float(np.abs(brake.astype(np.int16) - hold.astype(np.int16)).mean())
        for brake, hold in zip(predictions["brake"], predictions["hold"])
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    panels = list(comparison_panels(predictions, cases, times))
    with imageio.get_writer(
        args.output / "comparison.mp4", format="FFMPEG", fps=fps, codec="libx264",
        pixelformat="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart"]
    ) as video:
        for panel in panels:
            video.append_data(np.asarray(panel))
    selected = [panels[i] for i, t in enumerate(times) if float(t).is_integer()]
    contact_sheet = Image.new("RGB", (panels[0].width, panels[0].height * len(selected)))
    for i, panel in enumerate(selected):
        contact_sheet.paste(panel, (0, i * panel.height))
    contact_sheet.save(args.output / "comparison.jpg", quality=91, optimize=True)
    result = {
        "scene_token": scene["scene_token"],
        "split": scene["split"],
        "camera": scene["camera"],
        "history_sha256": scene["history_sha256"],
        "seed": args.seed,
        "conditioning_frames": config.condition_frames,
        "dit_steps_per_frame": config.dit_steps,
        "image_size_hw": list(config.image_size),
        "fps": fps,
        "times_s": times.tolist(),
        "initial_speed_mps": scene["initial_speed_mps"],
        "checkpoint_tensor_counts": checkpoint_report,
        "cases": cases,
        "mean_abs_rgb_delta_0_to_255_by_frame": difference,
        "matched_cuda_rng_after_rollouts": True,
        "pytorch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
