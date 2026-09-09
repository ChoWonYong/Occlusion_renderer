"""Download the demo, extract one person, and paste it onto KITTI clips."""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "work"
DRIVE_ID = "1GOwwfLCQGObRzSRQRyBx_3shXgajgx9y"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from common.config import config_path, load_config
from common.gpu_budget import enforce_account_gpu_budget
from common.io_video import group_annotations_by_image, group_frames_by_video
from pool.crops import bbox_iou
from synth.event_pipeline import (
    SceneBudget, _boxes_by_position, _frames_annotations, _load_rgb,
    _load_rgba, _resolve_placement, _victim_positions,
)
from synth.geometry import AlphaIntegralCache
from synth.scheduler import EventPlan, PoolTracklet, select_eligible_victims
from synth.tracklet_compositor import TrackletLayer, composite_tracklet_layers


def download(destination):
    if destination.exists():
        return
    print("Downloading demo_1.mp4", flush=True)
    url = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
    partial = destination.with_suffix(".part")
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        capture = cv2.VideoCapture(str(partial))
        valid, _ = capture.read()
        capture.release()
        if not valid:
            raise RuntimeError("Google Drive did not return a video; check the source link's sharing settings")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract(source, config, config_file):
    from segment.sam3 import Sam3Backend

    metadata = WORK / "person.json"
    if metadata.exists():
        return json.loads(metadata.read_text())
    enforce_account_gpu_budget(config["resources"], project_limit_key="phase1_sam_max_gpus")
    print("Loading SAM3; prompt=person", flush=True)
    backend = Sam3Backend(
        config_path(config, config_file, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, config_file, "paths", "sam3_bpe"),
        confidence_threshold=0.65,
    )
    capture = cv2.VideoCapture(str(source))
    fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps < 10 or total == 0:
        capture.release()
        raise RuntimeError("The source must be a readable video with at least 10 fps")
    sampled = {round(offset * fps / 10) for offset in range(int(total / fps * 10))}
    current, longest = [], []
    try:
        for index in range(total):
            success, bgr = capture.read()
            if not success:
                raise RuntimeError(f"Cannot decode source frame {index}")
            if index not in sampled:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            candidates = []
            for instance in backend.detect_and_mask(rgb, ["person"]):
                mask = np.asarray(instance.mask, dtype=np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                if count < 2:
                    continue
                component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                left, top, box_width, box_height, area = map(int, stats[component])
                if area < 500 or box_height < 100 or min(left, top) < 3:
                    continue
                if left + box_width > width - 3 or top + box_height > height - 3:
                    continue
                candidates.append((area, [left, top, box_width, box_height], labels == component))
            if not candidates:
                current = []
                continue
            _, box, mask = max(candidates, key=lambda item: item[0])
            if current and bbox_iou(current[-1]["crop_bbox_xywh"], box) < 0.25:
                current = []
            left, top, box_width, box_height = box
            rgba = np.dstack((rgb, mask.astype(np.uint8) * 255))
            filename = f"{index:06d}.png"
            Image.fromarray(rgba[top:top + box_height, left:left + box_width]).save(WORK / filename)
            current.append({"file_name": filename, "frame_index": index,
                            "crop_bbox_xywh": box, "source_image_size": [width, height]})
            if len(current) > len(longest):
                longest = list(current)
            if index % 30 == 0:
                print(f"SAM3: {index + 1}/{total} source frames", flush=True)
    finally:
        capture.release()
    if len(longest) < 30:
        raise RuntimeError("No continuous person tracklet of at least 30 frames found")
    metadata.write_text(json.dumps(longest, indent=2))
    print(f"Extracted {len(longest)} person frames at 10 fps", flush=True)
    return longest


def write_video(path, frames):
    height, width = frames[0].shape[:2]
    size = (width + width % 2, height + height % 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    try:
        for frame in frames:
            padded = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
            writer.write(cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def render(records, config, config_file, count):
    from data.kitti_tracking import convert_tracking_to_video_coco

    for record in records:
        record["rgba_path"] = str(WORK / record["file_name"])
    tracklet = PoolTracklet(1, "person", "demo_1", len(records), records, ("demo_1", "person"))
    settings = dict(config["tracklet_synthesis"], placement_draws=128, paste_jitter={"mode": "off"})
    dataset = convert_tracking_to_video_coco(
        config_path(config, config_file, "paths", "kitti_tracking"),
        config["split"]["train_sequences"], config["classes"]["kitti_map"],
    )
    annotations_by_image = group_annotations_by_image(dataset)
    categories = {category["id"]: category["name"] for category in dataset["categories"]}
    names = {video["id"]: video["name"] for video in dataset["videos"]}
    output = HERE / "visualization_tmp"
    output.mkdir(exist_ok=True)
    rng, cache, completed = random.Random(42), AlphaIntegralCache(), 0
    for video_id, frames in sorted(group_frames_by_video(dataset).items()):
        annotations = _frames_annotations(frames, annotations_by_image)
        scene = SceneBudget(_boxes_by_position(annotations), settings["peak_rho_max"])
        victims = select_eligible_victims(
            annotations, min_area=settings["victim_min_area"],
            max_base_occlusion=settings["victim_max_base_occlusion"], min_presence_frames=8,
        )
        rng.shuffle(victims)
        for victim in victims:
            exposure = rng.randint(30, min(tracklet.length, 80))
            event = EventPlan("victim", tracklet, exposure, victim.track_id, victim.category_id)
            resolved = _resolve_placement(
                event, _victim_positions(annotations, victim.track_id), categories[victim.category_id],
                {}, settings, (frames[0]["width"], frames[0]["height"]), len(frames),
                rng, cache, {}, set(), scene.accepts, None,
            )
            if resolved is None:
                continue
            _, placement = resolved
            originals, pasted, visible = [], [], []
            for offset, box in enumerate(placement["occluder_boxes"]):
                original = _load_rgb(frames[placement["start_position"] + offset]["source_path"])
                layer = TrackletLayer(1, _load_rgba(records[offset]["rgba_path"]), box)
                composite, rendered = composite_tracklet_layers(original, [layer], blend_method="none")
                originals.append(original)
                pasted.append(composite)
                if rendered:
                    visible.append(offset)
            if len(visible) < 30 or visible != list(range(visible[0], visible[-1] + 1)):
                continue
            first, last = visible[0], visible[-1]
            stem = f"person_{completed + 1:02d}_kitti{names[video_id]}"
            write_video(output / f"{stem}_original.mp4", originals[first:last + 1])
            write_video(output / f"{stem}_pasted.mp4", pasted[first:last + 1])
            completed += 1
            print(f"Saved {completed}/{count}: {stem} ({len(visible)} frames)", flush=True)
            break
        if completed == count:
            print(f"Done: {output}", flush=True)
            return
    raise RuntimeError(f"Only {completed}/{count} pairs could be placed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="number of pasted/original pairs (default: 1)")
    args = parser.parse_args()
    config, config_file = load_config(ROOT / "configs/default.yaml")
    if not 1 <= args.count <= len(config["split"]["train_sequences"]):
        parser.error("--count must be between 1 and the number of KITTI train sequences")
    for variable in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        directory = WORK / "cache" / variable.lower()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    source = HERE / "demo_1.mp4"
    download(source)
    records = extract(source, config, config_file)
    render(records, config, config_file, args.count)


if __name__ == "__main__":
    main()
