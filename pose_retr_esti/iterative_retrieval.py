"""
Iterative pose-based retrieval driver.

Usage:
    python -m _scripts.iterative_retrieval <input_image> <run_id>

Example:
    python -m _scripts.iterative_retrieval ./_samples/megumin.png 1

This script:
    - Runs pose-based retrieval on the input image (Stage 1).
    - Filters neighbors by distance / rank rules.
    - Downloads selected Danbooru images.
    - Repeats retrieval on the newly downloaded images (Stage 2, Stage 3, ...),
      using thresholds based on the *first* search rank.
    - Stops when a stage produces no new images.
"""

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Set, Optional

# 你的修改版 pose retrieval
from _scripts.pose_retr_mod import run_single_retrieval


# ======================= Data structures =======================

@dataclass
class RetrievedImage:
    """Metadata for one downloaded image in the iterative retrieval process."""
    stage: int
    post_id: int
    distance: float
    rank_in_search: int          # rank within the search that selected this image
    root_rank: int               # rank in the *first* search (Stage 1)
    parent_post_id: Optional[int]
    parent_stage: Optional[int]
    threshold_used: float
    file_path: str               # relative path to the saved image file


# ======================= Paths & CSV helpers =======================

def get_run_dir(run_id: int) -> Path:
    """
    Return the directory for this run, e.g. 'retrieve3' for run_id=3.
    Directory is created if it doesn't exist.
    """
    run_dir = Path(f"retrieve{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_log_path(run_dir: Path) -> Path:
    """Return the CSV log path inside a run directory."""
    return run_dir / "retrieval_log.csv"


def init_log_if_needed(log_path: Path) -> None:
    """
    Create CSV file with header if it does not exist yet.
    """
    if log_path.exists():
        return
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "post_id",
                "distance",
                "rank_in_search",
                "root_rank",
                "parent_post_id",
                "parent_stage",
                "threshold_used",
                "file_path",
            ],
        )
        writer.writeheader()


def append_log_rows(log_path: Path, rows: List[RetrievedImage]) -> None:
    """
    Append a list of RetrievedImage entries to the CSV log.
    """
    if not rows:
        return
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "post_id",
                "distance",
                "rank_in_search",
                "root_rank",
                "parent_post_id",
                "parent_stage",
                "threshold_used",
                "file_path",
            ],
        )
        for row in rows:
            writer.writerow(asdict(row))


def load_seen_ids_from_log(log_path: Path) -> Set[int]:
    """
    Read the log CSV (if exists) and return a set of post_ids that have already been recorded.
    """
    seen: Set[int] = set()
    if not log_path.exists():
        return seen
    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add(int(row["post_id"]))
            except (KeyError, ValueError):
                continue
    return seen


# ======================= Download helper (stub) =======================
import os
import requests  # ← 確認檔案最上面有這行，如果沒有要自己加

def download_post_image(post_id: int, out_dir: Path) -> Path:
    """
    Download a Danbooru post image by ID into out_dir and return the saved file path.

    Steps:
        1. GET https://danbooru.donmai.us/posts/<id>.json
        2. Read the 'file_url' field.
        3. Download that URL and save it as '<post_id>.<ext>' in out_dir.

    Note:
        - 如果環境沒網路，這函式會 raise requests 相關的錯誤。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 取得 post metadata
    api_url = f"https://danbooru.donmai.us/posts/{post_id}.json"
    r = requests.get(api_url)
    r.raise_for_status()
    data = r.json()

    file_url = data.get("file_url")
    if not file_url:
        # 有些 post 可能沒有實際檔案（被刪除或 restricted）
        raise RuntimeError(f"No file_url for post_id={post_id}")

    # 2. 決定輸出檔名（用 post_id + 副檔名）
    _, ext = os.path.splitext(file_url)
    if not ext:
        ext = ".jpg"
    filename = f"{post_id}{ext}"
    out_path = out_dir / filename

    # 3. 下載圖片
    img_resp = requests.get(file_url)
    img_resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(img_resp.content)

    return out_path



# ======================= Stage runners =======================

def run_stage1(
    input_image: Path,
    run_dir: Path,
    log_path: Path,
    seen_ids: Set[int],
) -> List[RetrievedImage]:
    """
    Stage 1:
        - Run pose retrieval on the original input image.
        - Take all neighbors with distance < 200.
        - Download them and record to log (log 實際寫入在 pipeline 那邊做).
    Returns:
        A list of RetrievedImage entries created in this stage.
    """
    # 1. 單張圖片做 pose-based retrieval
    result = run_single_retrieval(str(input_image))

    post_ids = result["neighbors_post_ids"]      # 長度 31
    distances = result["neighbors_distances"]    # 同長度

    # 2. (post_id, distance) 配對後，依 distance 由小排到大
    neighbors = list(zip(post_ids, distances))
    neighbors.sort(key=lambda x: x[1])

    stage = 1
    threshold = 180.0



    new_entries: List[RetrievedImage] = []

    # 3. 逐一檢查 31 個鄰居
    for rank, (post_id, dist) in enumerate(neighbors, start=1):
        dist_val = float(dist)

        # 3-1. 距離過大就不要
        if dist_val >= threshold:
            continue

        # 3-2. 已經看過的 id 不重複加
        if post_id in seen_ids:
            continue

        # 3-3. 下載這張圖到 run_dir 底下
        try:
            saved_path = download_post_image(int(post_id), run_dir)
        except Exception as e:
            print(f"[Stage 1] Failed to download post {post_id}: {e}")
            continue

        # 3-4. 建立一筆 metadata
        entry = RetrievedImage(
            stage=stage,
            post_id=int(post_id),
            distance=dist_val,
            rank_in_search=rank,      # 這次搜尋裡的名次
            root_rank=rank,           # 第一次搜尋，root_rank 就等於自己的 rank
            parent_post_id=None,      # Stage 1 沒 parent
            parent_stage=None,
            threshold_used=threshold,
            file_path=str(saved_path),  # 先存絕對或相對路徑都可
        )

        new_entries.append(entry)
        seen_ids.add(int(post_id))

    # 不在這裡寫 log，讓 pipeline 統一呼叫 append_log_rows
    return new_entries



def run_stage2(
    stage1_entries: List[RetrievedImage],
    run_dir: Path,
    log_path: Path,
    seen_ids: Set[int],
) -> List[RetrievedImage]:
    """
    Stage 2:
        For each image from Stage 1:
            - Use its *local file* as query.
            - Run pose retrieval (31 neighbors).
            - For this query image, threshold = round(200 - 6.45 * root_rank).
            - Keep neighbors with distance < threshold and not in seen_ids.
            - Download & log.
    Returns:
        A list of newly created RetrievedImage entries (Stage 2).
    """
    stage = 2
    new_entries: List[RetrievedImage] = []

    for parent in stage1_entries:
        query_path = Path(parent.file_path)

        # 沒找到檔案就跳過，避免之前下載失敗之類的狀況
        if not query_path.is_file():
            print(f"[Stage 2] Warning: query image not found: {query_path}")
            continue

        # 1. 用這張 Stage1 的圖片做一次 retrieval
        result = run_single_retrieval(str(query_path))
        post_ids = result["neighbors_post_ids"]
        distances = result["neighbors_distances"]

        neighbors = list(zip(post_ids, distances))
        neighbors.sort(key=lambda x: x[1])  # 依 distance 由小到大排序

        # 2. 用「第一次搜尋的 rank」算 threshold2
        root_rank = parent.root_rank
        threshold2 = round(180.0 - 6.0 * root_rank)

        # 如果門檻 <= 0，這個 parent 基本上不可能通過任何圖，直接跳過
        if threshold2 <= 0:
            continue

        # 3. 逐一檢查 31 個 neighbors
        for rank, (post_id, dist) in enumerate(neighbors, start=1):
            dist_val = float(dist)

            # 距離要小於 threshold2
            if dist_val >= threshold2:
                continue

            # 不要重複下載已經出現過的 id
            if post_id in seen_ids:
                continue

            # 嘗試下載這張圖，如果失敗就跳過
            try:
                saved_path = download_post_image(int(post_id), run_dir)
            except Exception as e:
                print(f"[Stage 2] Failed to download post {post_id}: {e}")
                continue

            entry = RetrievedImage(
                stage=stage,
                post_id=int(post_id),
                distance=dist_val,
                rank_in_search=rank,          # 這次 Stage2 搜尋中的名次
                root_rank=root_rank,          # 一路沿用 Stage1 的 root_rank
                parent_post_id=parent.post_id,
                parent_stage=parent.stage,
                threshold_used=float(threshold2),
                file_path=str(saved_path),
            )

            new_entries.append(entry)
            seen_ids.add(int(post_id))

    return new_entries



def run_stage3(
    stage2_entries: List[RetrievedImage],
    run_dir: Path,
    log_path: Path,
    seen_ids: Set[int],
) -> List[RetrievedImage]:
    """
    Stage 3:
        For each image from Stage 2:
            - Use its *local file* as query.
            - Run pose retrieval (31 neighbors).
            - Threshold = ceil(200 - (6.45**2) * root_rank) based on root_rank from Stage 1.
            - Keep neighbors with distance < threshold and not in seen_ids.
            - Download & log.
        Pipeline 在外層會檢查：如果本 stage 新增 0 張，就停止。
    Returns:
        A list of newly created RetrievedImage entries (Stage 3).
    """
    import math

    stage = 3
    new_entries: List[RetrievedImage] = []

    for parent in stage2_entries:
        query_path = Path(parent.file_path)
        if not query_path.is_file():
            print(f"[Stage 3] Warning: query image not found: {query_path}")
            continue

        # 1. 用這張 Stage2 圖片做 retrieval
        result = run_single_retrieval(str(query_path))
        post_ids = result["neighbors_post_ids"]
        distances = result["neighbors_distances"]

        neighbors = list(zip(post_ids, distances))
        neighbors.sort(key=lambda x: x[1])  # 依 distance 排序

        # 2. 用第一次搜尋的 root_rank 算 threshold3
        root_rank = parent.root_rank
        threshold3 = math.ceil(180.0 - (6.0 ** 2) * root_rank)

        if threshold3 <= 0:
            continue

        # 3. 掃 31 個 neighbors
        for rank, (post_id, dist) in enumerate(neighbors, start=1):
            dist_val = float(dist)

            if dist_val >= threshold3:
                continue
            if post_id in seen_ids:
                continue

            try:
                saved_path = download_post_image(int(post_id), run_dir)
            except Exception as e:
                print(f"[Stage 3] Failed to download post {post_id}: {e}")
                continue

            entry = RetrievedImage(
                stage=stage,
                post_id=int(post_id),
                distance=dist_val,
                rank_in_search=rank,          # Stage3 這次搜尋中的名次
                root_rank=root_rank,          # 一直沿用 Stage1 的 rank
                parent_post_id=parent.post_id,
                parent_stage=parent.stage,
                threshold_used=float(threshold3),
                file_path=str(saved_path),
            )

            new_entries.append(entry)
            seen_ids.add(int(post_id))

    return new_entries



# ======================= Orchestration (main pipeline) =======================

def iterative_retrieval_pipeline(input_image: Path, run_id: int) -> None:
    """
    Full pipeline:
        - Prepare run directory and log.
        - Run Stage 1.
        - Run Stage 2 using Stage 1 results.
        - Run Stage 3 using Stage 2 results.
        - Stop if Stage 3 creates zero new images.
    """
    run_dir = get_run_dir(run_id)
    log_path = get_log_path(run_dir)
    init_log_if_needed(log_path)

    # Load already-seen IDs (useful if re-running)
    seen_ids = load_seen_ids_from_log(log_path)

    # Stage 1
    print("[Stage 1] Running retrieval for input image...")
    stage1_entries = run_stage1(input_image, run_dir, log_path, seen_ids)
    append_log_rows(log_path, stage1_entries)
    print(f"[Stage 1] New images: {len(stage1_entries)}")

    # Stage 2
    print("[Stage 2] Expanding from Stage 1 images...")
    stage2_entries = run_stage2(stage1_entries, run_dir, log_path, seen_ids)
    append_log_rows(log_path, stage2_entries)
    print(f"[Stage 2] New images: {len(stage2_entries)}")

    # Stage 3
    print("[Stage 3] Expanding from Stage 2 images...")
    stage3_entries = run_stage3(stage2_entries, run_dir, log_path, seen_ids)
    append_log_rows(log_path, stage3_entries)
    print(f"[Stage 3] New images: {len(stage3_entries)}")

    # Stopping condition: if Stage 3 adds nothing, we stop.
    if len(stage3_entries) == 0:
        print("[Pipeline] Stage 3 produced no new images. Stopping.")
    else:
        print("[Pipeline] Stage 3 produced new images. "
              "Further stages are not yet implemented.")


# ======================= CLI entry =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_img", help="Path to the input image.")
    parser.add_argument("run_id", type=int, help="Run identifier, e.g. 1 -> 'retrieve1' directory.")
    args = parser.parse_args()

    input_image = Path(args.fn_img)
    if not input_image.is_file():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    iterative_retrieval_pipeline(input_image, args.run_id)


if __name__ == "__main__":
    main()
