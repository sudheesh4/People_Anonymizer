import argparse
import copy
import math
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    Sam2Processor,
    Sam2Model,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def read_video(path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # BGR uint8
    cap.release()
    return frames, float(fps), (W, H)

def bgr_to_pil(batch_bgr: List[np.ndarray]) -> List[Image.Image]:
    imgs = []
    for b in batch_bgr:
        rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(rgb))
    return imgs

def write_video_rgb(path: str, rgb_frames: List[np.ndarray], fps: float) -> None:
    if not rgb_frames:
        raise ValueError("No frames to write.")
    H, W, _ = rgb_frames[0].shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for rgb in rgb_frames:
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    vw.release()

# ---------------------- Models ----------------------
def load_models(detr_dir: str , sam2_dir: str ):
    """
    Load DETR + SAM2 either from local directories (if provided) or download from internet.
    """
    if detr_dir:
        detr_proc = DetrImageProcessor.from_pretrained(detr_dir, local_files_only=True)
        detr = DetrForObjectDetection.from_pretrained(detr_dir, local_files_only=True).to(DEVICE).eval()
    else:
        detr_proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE).eval()

    if sam2_dir:
        sam2_proc = Sam2Processor.from_pretrained(sam2_dir, local_files_only=True)
        sam2 = Sam2Model.from_pretrained(sam2_dir, local_files_only=True).to(DEVICE).eval()
    else:
        sam2_proc = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
        sam2 = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(DEVICE).eval()

    return detr_proc, detr, sam2_proc, sam2

# ---------------------- Mask utilities ----------------------
def to_hw_bool(mask) -> np.ndarray:
    """Accepts (H,W), (H,W,1) or (1,H,W). Returns boolean (H,W)."""
    m = np.array(mask)
    if m.ndim == 3 and m.shape[-1] == 1:   # (H,W,1)
        m = m[..., 0]
    elif m.ndim == 3 and m.shape[0] == 1:  # (1,H,W)
        m = m[0]
    return (m > 0).astype(bool)

def distinct_colors_rgb(n, seed=123) -> List[tuple]:
    rng = np.random.default_rng(seed)
    cols = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
    return [tuple(map(int, c)) for c in cols]

def compose_masked_rgb(rgb_img: np.ndarray, masks_bool: List[np.ndarray]) -> np.ndarray:
    """
    rgb_img: (H,W,3) uint8, RGB
    masks_bool: list of (H,W) boolean masks
    Replace ONLY person pixels with distinct solid colors; background unchanged.
    """
    if not masks_bool:
        return rgb_img

    out = rgb_img.copy()
    H, W, _ = out.shape
    colors = distinct_colors_rgb(len(masks_bool), seed=len(masks_bool) + H + W)

    for color, m in zip(colors, masks_bool):
        m = m.astype(bool)
        if not m.any():
            continue
        out[m] = color


    return out

# ---------------------- Core pipeline ----------------------
def person_boxes_from_detr(
    pil_batch: List[Image.Image],
    detr_proc: DetrImageProcessor,
    detr: DetrForObjectDetection,
    conf_thresh: float,
    max_persons_per_frame: int
) -> List[List[List[float]]]:
    """
    Returns list (per image) of [x1,y1,x2,y2] person boxes in pixel coords.
    """
    with torch.no_grad():
        inputs = detr_proc(images=pil_batch, return_tensors="pt").to(DEVICE)
        outputs = detr(**inputs)
        target_sizes = torch.tensor([img.size[::-1] for img in pil_batch], device=DEVICE)  # (H,W)
        results = detr_proc.post_process_object_detection(outputs, threshold=conf_thresh, target_sizes=target_sizes)

    id2label = detr.config.id2label
    person_boxes_per_image: List[List[List[float]]] = []
    for res in results:
        boxes = []
        for score, label_id, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            label = id2label.get(label_id, str(label_id))
            if label == "person":
                boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                if len(boxes) >= max_persons_per_frame:
                    break
        person_boxes_per_image.append(boxes)
    return person_boxes_per_image

def sam2_masks_for_boxes(
    img: Image.Image,
    img_boxes: List[List[float]],
    sam2_proc: Sam2Processor,
    sam2: Sam2Model
) -> List[np.ndarray]:
    """
    Runs SAM2 on a single image with a list of boxes. Returns list of masks as (H,W) or (H,W,1).
    """
    if not img_boxes:
        return []

    inputs = sam2_proc(images=img, input_boxes=[img_boxes], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = sam2(**inputs, multimask_output=False)

    # outputs.pred_masks shape: (batch=1, num_boxes, channels=1 or 3?, 256, 256) depending on model
    masks_pt = torch.nn.functional.interpolate(
        outputs.pred_masks[0].float(),
        size=(img.height, img.width),
        mode="bilinear",
        align_corners=False
    )
    masks_np = (masks_pt.cpu().numpy() > 0.5)  # (num_boxes, C, H, W)

    out_masks: List[np.ndarray] = []
    for j in range(masks_np.shape[0]):
        m = masks_np[j]
        # collapse channel dimension if present
        if m.ndim == 3:
            if m.shape[0] == 1:      # (1,H,W) -> (H,W)
                m = m[0]
            elif m.shape[-1] == 1:   # (H,W,1) -> (H,W)
                m = m[..., 0]
            else:
                # If multiple channels, OR them
                m = np.any(m, axis=0)
        out_masks.append(m.astype(bool))
    return out_masks

def process_video(
    in_path: Path,
    out_path: Path,
    detr_proc: DetrImageProcessor,
    detr: DetrForObjectDetection,
    sam2_proc: Sam2Processor,
    sam2: Sam2Model,
    batch_size: int,
    conf_thresh: float,
    max_persons_per_frame: int
) -> None:
    print(f">>>> Reading video: {in_path}")
    frames_bgr, fps, _ = read_video(str(in_path))
    print(f">>>> {len(frames_bgr)} frames @ {fps:.2f} FPS")

    masked_rgb_frames: List[np.ndarray] = []
    num_batches = math.ceil(len(frames_bgr) / batch_size)

    for bi in range(num_batches):
        s = bi * batch_size
        e = min((bi + 1) * batch_size, len(frames_bgr))
        batch_bgr = frames_bgr[s:e]
        pil_batch = bgr_to_pil(batch_bgr)

        # DETR person boxes
        person_boxes_per_image = person_boxes_from_detr(
            pil_batch, detr_proc, detr, conf_thresh=conf_thresh, max_persons_per_frame=max_persons_per_frame
        )

        # SAM2 masks + compose colored output
        for j, (img, boxes) in enumerate(zip(pil_batch, person_boxes_per_image)):
            masks = sam2_masks_for_boxes(img, boxes, sam2_proc, sam2)
            rgb = np.array(img)  # (H,W,3)
            comp = compose_masked_rgb(rgb, [to_hw_bool(m) for m in masks])
            masked_rgb_frames.append(comp)

        print(f"  batch {bi+1}/{num_batches} processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f">>>> Writing: {out_path}")
    write_video_rgb(str(out_path), masked_rgb_frames, fps)

# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Person segmentation in videos using DETR + SAM2")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to a single input video")
    p.add_argument("--out-dir", type=str, default=".", help="Directory to save outputs (default: current directory)")
    p.add_argument("--detr-dir", type=str, default=None, help="Local DETR model directory (optional)")
    p.add_argument("--sam2-dir", type=str, default=None, help="Local SAM2 model directory (optional)")
    p.add_argument("--batch-size", type=int, default=4, help="Frames per batch (default: 4)")
    p.add_argument("--detr-thresh", type=float, default=0.7, help="DETR confidence threshold for 'person' (default: 0.7)")
    p.add_argument("--max-persons-per-frame", type=int, default=20, help="Guardrail on number of persons per frame")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(">>>> LOADING MODELS")
    detr_proc, detr, sam2_proc, sam2 = load_models(args.detr_dir, args.sam2_dir)
    print(f"Device: {DEVICE}")

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(f"--input not found: {in_path}")
        out_path = out_dir / f"{in_path.stem}_person_segments.mp4"
        process_video(
            in_path, out_path, detr_proc, detr, sam2_proc, sam2,
            batch_size=args.batch_size, conf_thresh=args.detr_thresh, max_persons_per_frame=args.max_persons_per_frame
        )
    else:
        print(">>>>INPUT VIDEO PATH REQUIRED (--input <path-to-video>) ")

if __name__ == "__main__":
    main()
