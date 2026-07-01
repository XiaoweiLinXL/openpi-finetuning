#!/usr/bin/env python3
"""Convert AgiBot G1 raw records to LeRobot v2.1 format."""

import json
import pathlib
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

RAW_DIR = pathlib.Path.home() / "agibot_g1_data/data/record"
REPO_ID = "local/agibot_g1_omnipicker"
TARGET_FPS = 30
IMAGE_H, IMAGE_W = 480, 640
STATE_DIM = 18  # 14 joints + 2 waist + 1 left_eff + 1 right_eff
ACTION_DIM = 18

STATE_NAMES = (
    [f"joint_{i}" for i in range(14)]
    + ["waist_0", "waist_1"]
    + ["left_effector", "right_effector"]
)

FEATURES = {
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": (IMAGE_H, IMAGE_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": (IMAGE_H, IMAGE_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.head_color": {
        "dtype": "video",
        "shape": (IMAGE_H, IMAGE_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": STATE_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": STATE_NAMES,
    },
}

TASK = "pick and place objects using the omnipicker"


def nearest_idx(timestamps: np.ndarray, target: int) -> int:
    return int(np.argmin(np.abs(timestamps - target)))


def find_valid_episodes(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    valid = []
    for meta_path in sorted(raw_dir.glob("*/meta_info.json")):
        m = json.loads(meta_path.read_text())
        if not (
            (m.get("data_validate") or {}).get("validate") is True
            and (m.get("integrity") or {}).get("integrity") is True
            and m.get("fps_validate") is True
            and (m.get("file_size") or 0) > 0
        ):
            continue
        ep_dir = meta_path.parent
        # Skip episodes where HDF5 action data is actually empty (metadata can lie)
        try:
            with h5py.File(ep_dir / "record/raw_joints.h5", "r") as f:
                if f["action/joint/timestamp"].shape[0] == 0:
                    print(f"  Skipping {ep_dir.name[:8]}: empty action data")
                    continue
        except Exception:
            continue
        valid.append(ep_dir)
    return sorted(valid)


def decode_h265_frames(
    h265_path: pathlib.Path, ts_txt_path: pathlib.Path, out_dir: pathlib.Path
) -> np.ndarray:
    lines = ts_txt_path.read_text().strip().split("\n")
    timestamps = np.array([int(ln.split()[0]) for ln in lines], dtype=np.int64)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "hevc", "-i", str(h265_path),
         "-q:v", "2", str(out_dir / "%06d.jpg")],
        check=True,
    )
    return timestamps


def load_image_resized(path: pathlib.Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def process_episode(ep_dir: pathlib.Path, dataset: LeRobotDataset) -> int:
    with h5py.File(ep_dir / "record/raw_joints.h5", "r") as f:
        s_joint_pos = f["state/joint/position"][()]
        s_joint_ts  = f["state/joint/timestamp"][()]
        s_waist_pos = f["state/waist/position"][()]
        s_waist_ts  = f["state/waist/timestamp"][()]
        s_left_eff  = f["state/left_effector/position"][()]
        s_right_eff = f["state/right_effector/position"][()]

        a_joint_pos = f["action/joint/position"][()]
        a_joint_ts  = f["action/joint/timestamp"][()]
        a_waist_pos = f["action/waist/position"][()]
        a_waist_ts  = f["action/waist/timestamp"][()]
        a_left_eff  = f["action/left_effector/position"][()]
        a_left_ts   = f["action/left_effector/timestamp"][()]
        a_right_eff = f["action/right_effector/position"][()]
        a_right_ts  = f["action/right_effector/timestamp"][()]

    hand_left_dir   = ep_dir / "camera/hand_left/color"
    hand_right_dir  = ep_dir / "camera/hand_right/color"
    hand_left_imgs  = sorted(hand_left_dir.glob("*.jpg"))
    hand_right_imgs = sorted(hand_right_dir.glob("*.jpg"))
    hand_left_ts    = np.array([int(p.stem) for p in hand_left_imgs], dtype=np.int64)
    hand_right_ts   = np.array([int(p.stem) for p in hand_right_imgs], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        head_ts = decode_h265_frames(
            ep_dir / "camera/head_color/head_color.h265",
            ep_dir / "camera/head_color/head_color.txt",
            tmp_path,
        )
        head_frames = sorted(tmp_path.glob("*.jpg"))

        t_start = max(
            s_joint_ts[0], a_joint_ts[0],
            hand_left_ts[0], hand_right_ts[0], head_ts[0],
        )
        t_end = min(
            s_joint_ts[-1], a_joint_ts[-1],
            hand_left_ts[-1], hand_right_ts[-1], head_ts[-1],
        )

        dt_ns = int(1e9 / TARGET_FPS)
        grid = np.arange(t_start, t_end, dt_ns, dtype=np.int64)
        print(f"  {ep_dir.name[:8]}... → {len(grid)} frames at {TARGET_FPS}Hz")

        for t in grid:
            ji = nearest_idx(s_joint_ts, t)
            wi = nearest_idx(s_waist_ts, t)
            state_vec = np.concatenate([
                s_joint_pos[ji],
                s_waist_pos[wi],
                [s_left_eff[ji, 0] / 120.0],
                [s_right_eff[ji, 0] / 120.0],
            ]).astype(np.float32)

            aj = nearest_idx(a_joint_ts, t)
            aw = nearest_idx(a_waist_ts, t)
            al = nearest_idx(a_left_ts, t)
            ar = nearest_idx(a_right_ts, t)
            action_vec = np.concatenate([
                a_joint_pos[aj],
                a_waist_pos[aw],
                a_left_eff[al],
                a_right_eff[ar],
            ]).astype(np.float32)

            dataset.add_frame({
                "observation.state": state_vec,
                "action": action_vec,
                "observation.images.hand_left":  load_image_resized(hand_left_imgs[nearest_idx(hand_left_ts, t)]),
                "observation.images.hand_right": load_image_resized(hand_right_imgs[nearest_idx(hand_right_ts, t)]),
                "observation.images.head_color": load_image_resized(head_frames[nearest_idx(head_ts, t)]),
            }, task=TASK)

        dataset.save_episode()
        return len(grid)


def main():
    test_only = "--test" in sys.argv
    episodes = find_valid_episodes(RAW_DIR)
    if test_only:
        episodes = episodes[:1]
    print(f"Found {len(episodes)} valid episodes (test={test_only})")

    # Resume support: detect existing dataset and skip already-done episodes
    dataset_root = pathlib.Path.home() / ".cache/huggingface/lerobot" / REPO_ID
    if not test_only and dataset_root.exists() and (dataset_root / "meta/info.json").exists():
        dataset = LeRobotDataset(repo_id=REPO_ID, root=dataset_root)
        dataset.start_image_writer(num_processes=2, num_threads=4)
        num_done = dataset.num_episodes
        print(f"Resuming: {num_done} episodes already done, skipping to episode {num_done + 1}")
        episodes = episodes[num_done:]
    else:
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=TARGET_FPS,
            robot_type="agibot-g1",
            features=FEATURES,
            image_writer_threads=4,
            image_writer_processes=2,
        )
        num_done = 0

    total_frames = 0
    for i, ep_dir in enumerate(episodes):
        print(f"[{num_done + i + 1}/{num_done + len(episodes)}] {ep_dir.name}")
        total_frames += process_episode(ep_dir, dataset)

    dataset.stop_image_writer()
    print(f"\nDone! {num_done + len(episodes)} episodes total, {total_frames} new frames")
    print(f"Dataset: {dataset.root}")


if __name__ == "__main__":
    main()
