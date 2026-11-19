"""Extract one NAVSIM mini scene's four CAM_F0 observations and ego poses."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def candidates(metadata_archive, names, scene_filter, requested_token):
    selected = set(scene_filter["tokens"])
    logs = set(scene_filter["log_names"]) & set(names)
    n_history = scene_filter["num_history_frames"]
    length = n_history + scene_filter["num_future_frames"]
    interval = scene_filter["frame_interval"] or length
    matches = []
    with tarfile.open(metadata_archive, "r|gz") as archive:
        for member in archive:
            name = Path(member.name).stem
            if not member.isfile() or name not in logs:
                continue
            # The source is the official OpenScene release archive.
            frames = pickle.load(archive.extractfile(member))
            logs.remove(name)
            for start in range(0, len(frames) - length + 1, interval):
                history = frames[start : start + n_history]
                anchor = history[-1]
                if anchor["token"] not in selected or not anchor["roadblock_ids"]:
                    continue
                if requested_token and anchor["token"] != requested_token:
                    continue
                transforms = np.repeat(np.eye(4)[None], n_history, axis=0)
                for frame, pose in zip(history, transforms):
                    w, x, y, z = frame["ego2global_rotation"]
                    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
                    pose[:3, 3] = frame["ego2global_translation"]
                local = np.linalg.inv(transforms[-1]) @ transforms
                yaw = Rotation.from_matrix(local[:, :3, :3]).as_euler("zyx", degrees=True)[:, 0]
                speed = float(anchor["ego_dynamic_state"][0])
                if not 5.0 <= speed <= 10.0 or np.max(np.abs(yaw)) > 6:
                    continue
                times_us = [int(frame["timestamp"]) for frame in history]
                if not np.allclose(np.diff(times_us), 500_000, atol=100_000):
                    continue
                camera_paths = [frame["cams"]["CAM_F0"]["data_path"] for frame in history]
                matches.append({
                    "scene_token": anchor["token"],
                    "log_name": name,
                    "split": "navmini",
                    "camera": "CAM_F0",
                    "camera_paths": camera_paths,
                    "history_tokens": [frame["token"] for frame in history],
                    "timestamps_us": times_us,
                    "history_transforms_local": local.tolist(),
                    "initial_speed_mps": speed,
                })
            if not logs:
                break
    return matches


def extract_first_complete(camera_archive, scenes, output):
    needed = {path for scene in scenes for path in scene["camera_paths"]}
    captured = {}
    marker = "/sensor_blobs/"
    try:
        with tarfile.open(camera_archive, "r|gz") as archive:
            for entry in archive:
                name = str(entry.name)
                if not entry.isfile() or marker not in name:
                    continue
                name = name.split(marker, 1)[1].removeprefix("mini/")
                if name not in needed:
                    continue
                captured[name] = archive.extractfile(entry).read()
                for scene in scenes:
                    if not all(path in captured for path in scene["camera_paths"]):
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    scene["history_files"] = []
                    scene["history_sha256"] = []
                    for i, path in enumerate(scene["camera_paths"]):
                        filename = f"history_{i:02}.jpg"
                        image = captured[path]
                        (output / filename).write_bytes(image)
                        scene["history_files"].append(filename)
                        scene["history_sha256"].append(sha256(image).hexdigest())
                    (output / "scene.json").write_text(
                        json.dumps(scene, indent=2) + "\n", encoding="utf-8", newline="\n"
                    )
                    return scene
    except (EOFError, tarfile.ReadError):
        print(f"Archive stream ended after finding {len(captured)} candidate images.")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned DreamerAD checkout")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--camera-archive", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-index-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", help="Extract this NAVSIM scene token")
    args = parser.parse_args()
    scene_filter_path = args.source / (
        "navsim/planning/script/config/common/train_test_split/scene_filter/navmini.yaml"
    )
    scene_filter = yaml.safe_load(scene_filter_path.read_text())
    index = json.loads(args.shard_index_file.read_text())
    names = index[f"openscene_sensor_mini_{args.shard_index}"]
    scenes = candidates(args.metadata, names, scene_filter, args.token)
    if not scenes:
        raise SystemExit("No matching NAVSIM scenes in this camera shard")
    print(f"Loaded {len(scenes)} candidate scenes from metadata.", flush=True)
    scene = extract_first_complete(args.camera_archive, scenes, args.output)
    if scene is None:
        raise SystemExit("No candidate has all four history frames in this camera archive")
    scene["camera_shard_index"] = args.shard_index
    (args.output / "scene.json").write_text(
        json.dumps(scene, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(scene["scene_token"], scene["log_name"], scene["initial_speed_mps"])


if __name__ == "__main__":
    main()
