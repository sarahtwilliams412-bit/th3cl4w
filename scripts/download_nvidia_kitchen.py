#!/usr/bin/env python3
"""Download NVIDIA PhysicalAI-Robotics-Kitchen-Sim-Demos dataset.

Selectively downloads task subsets most relevant to th3cl4w:
- Pick-and-place atomic tasks (17 tasks, directly relevant)
- Target tasks with 500 trajectories each (higher quality)

Usage:
    # Download pick-and-place tasks only (~5GB)
    python -m scripts.download_nvidia_kitchen --subset pick_place

    # Download all target tasks (~62GB)
    python -m scripts.download_nvidia_kitchen --subset target

    # Download everything (~175GB)
    python -m scripts.download_nvidia_kitchen --subset all

    # List available tasks
    python -m scripts.download_nvidia_kitchen --list
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "nvidia_kitchen"

# Task categories relevant to th3cl4w
PICK_PLACE_TASKS = [
    "pretrain/atomic/PickPlaceCounterToOven",
    "pretrain/atomic/PickPlaceCounterToBlender",
    "pretrain/atomic/PickPlaceCounterToCabinet",
    "pretrain/atomic/PickPlaceCounterToDrawer",
    "pretrain/atomic/PickPlaceCounterToMicrowave",
    "pretrain/atomic/PickPlaceCounterToSink",
    "pretrain/atomic/PickPlaceCounterToStandMixer",
    "pretrain/atomic/PickPlaceCounterToStove",
    "pretrain/atomic/PickPlaceCounterToToasterOven",
    "pretrain/atomic/PickPlaceCabinetToCounter",
    "pretrain/atomic/PickPlaceDrawerToCounter",
    "pretrain/atomic/PickPlaceFridgeDrawerToShelf",
    "pretrain/atomic/PickPlaceFridgeShelfToDrawer",
    "pretrain/atomic/PickPlaceMicrowaveToCounter",
    "pretrain/atomic/PickPlaceSinkToCounter",
    "pretrain/atomic/PickPlaceStoveToCounter",
    "pretrain/atomic/PickPlaceToasterOvenToCounter",
]

TARGET_ATOMIC_TASKS = [
    "target/atomic/CloseBlenderLid",
    "target/atomic/CloseFridge",
    "target/atomic/CloseToasterOvenDoor",
    "target/atomic/CoffeeSetupMug",
    "target/atomic/NavigateKitchen",
    "target/atomic/OpenCabinet",
    "target/atomic/OpenDrawer",
    "target/atomic/OpenStandMixerHead",
    "target/atomic/PickPlaceCounterToCabinet",
    "target/atomic/PickPlaceCounterToStove",
    "target/atomic/PickPlaceDrawerToCounter",
    "target/atomic/PickPlaceSinkToCounter",
    "target/atomic/PickPlaceToasterToCounter",
    "target/atomic/SlideDishwasherRack",
    "target/atomic/TurnOffStove",
    "target/atomic/TurnOnElectricKettle",
    "target/atomic/TurnOnMicrowave",
    "target/atomic/TurnOnSinkFaucet",
]

TARGET_COMPOSITE_TASKS = [
    "target/composite/ArrangeBreadBasket",
    "target/composite/ArrangeTea",
    "target/composite/GetToastedBread",
    "target/composite/HeatKebabSandwich",
    "target/composite/KettleBoiling",
    "target/composite/LoadDishwasher",
    "target/composite/MakeIceLemonade",
    "target/composite/PrepareCoffee",
    "target/composite/SearingMeat",
    "target/composite/SetUpCuttingStation",
    "target/composite/StackBowlsCabinet",
    "target/composite/SteamInMicrowave",
    "target/composite/StoreLeftoversInBowl",
    "target/composite/WashFruitColander",
]

SUBSETS = {
    "pick_place": PICK_PLACE_TASKS,
    "target_atomic": TARGET_ATOMIC_TASKS,
    "target_composite": TARGET_COMPOSITE_TASKS,
    "target": TARGET_ATOMIC_TASKS + TARGET_COMPOSITE_TASKS,
    "all": None,  # Download everything
}


def download_task(task_path: str, output_dir: Path, max_retries: int = 4):
    """Download a single task archive using huggingface-cli."""
    # Find the exact tar file path within the task directory
    # Convention: {task_path}/{YYYYMMDD}/lerobot.tar
    logger.info("Downloading: %s", task_path)

    for attempt in range(max_retries):
        try:
            cmd = [
                "huggingface-cli", "download",
                "--repo-type", "dataset",
                REPO_ID,
                "--include", f"{task_path}/*/lerobot.tar",
                "--local-dir", str(output_dir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info("Downloaded: %s", task_path)
                return True
            else:
                logger.warning(
                    "Download attempt %d failed for %s: %s",
                    attempt + 1, task_path, result.stderr[:200],
                )
        except subprocess.TimeoutExpired:
            logger.warning("Download timed out for %s (attempt %d)", task_path, attempt + 1)
        except FileNotFoundError:
            logger.error(
                "huggingface-cli not found. Install with: pip install huggingface_hub"
            )
            return False

        # Exponential backoff
        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            logger.info("Retrying in %ds...", wait)
            import time
            time.sleep(wait)

    logger.error("Failed to download %s after %d attempts", task_path, max_retries)
    return False


def extract_task(tar_path: Path, extract_dir: Path):
    """Extract a downloaded lerobot.tar archive."""
    if not tar_path.exists():
        return False

    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(extract_dir)],
            check=True, capture_output=True,
        )
        logger.info("Extracted: %s → %s", tar_path, extract_dir)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Extraction failed for %s: %s", tar_path, e.stderr[:200])
        return False


def list_tasks():
    """Print all available task subsets."""
    print("\n=== NVIDIA Kitchen-Sim-Demos Task Subsets ===\n")
    for subset_name, tasks in SUBSETS.items():
        if tasks is None:
            print(f"  {subset_name}: ALL tasks (~175GB, 367 tasks)")
        else:
            print(f"  {subset_name}: {len(tasks)} tasks")
            for t in tasks[:5]:
                print(f"    - {t}")
            if len(tasks) > 5:
                print(f"    ... and {len(tasks) - 5} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA Kitchen-Sim-Demos dataset"
    )
    parser.add_argument(
        "--subset", choices=list(SUBSETS.keys()), default="pick_place",
        help="Which subset to download (default: pick_place)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available task subsets and exit",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract tar archives after downloading",
    )
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = SUBSETS[args.subset]

    if tasks is None:
        logger.info("Downloading FULL dataset (~175GB) to %s", output_dir)
        cmd = [
            "huggingface-cli", "download",
            "--repo-type", "dataset",
            REPO_ID,
            "--local-dir", str(output_dir),
        ]
        subprocess.run(cmd)
        return

    logger.info(
        "Downloading %d tasks (%s subset) to %s",
        len(tasks), args.subset, output_dir,
    )

    success = 0
    failed = 0
    for task_path in tasks:
        if download_task(task_path, output_dir):
            success += 1
            if args.extract:
                # Find and extract the tar file
                for tar_file in output_dir.rglob(f"{task_path}/*/lerobot.tar"):
                    task_name = task_path.split("/")[-1]
                    extract_dir = output_dir / "extracted" / task_name
                    extract_task(tar_file, extract_dir)
        else:
            failed += 1

    logger.info(
        "Download complete: %d succeeded, %d failed out of %d tasks",
        success, failed, len(tasks),
    )


if __name__ == "__main__":
    main()
