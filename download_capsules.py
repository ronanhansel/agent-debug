#!/usr/bin/env python3
"""Download and extract all CoreBench capsules."""

import json
import os
import urllib.request
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
COREBENCH_DIR = "hal-harness/hal/benchmarks/corebench"
CAPSULES_DIR = os.path.join(COREBENCH_DIR, "capsules")
CORE_TEST_PATH = os.path.join(COREBENCH_DIR, "core_test.json")
BASE_URL = "https://corebench.cs.princeton.edu/capsules"

def download_and_extract(capsule_id, max_retries=3):
    """Download and extract a single capsule."""
    capsule_dir = os.path.join(CAPSULES_DIR, capsule_id)

    # Skip if already exists
    if os.path.exists(capsule_dir):
        return f"{capsule_id}: already exists, skipped"

    tar_path = os.path.join(CAPSULES_DIR, f"{capsule_id}.tar.gz")
    url = f"{BASE_URL}/{capsule_id}.tar.gz"

    for attempt in range(max_retries):
        try:
            # Download
            urllib.request.urlretrieve(url, tar_path)

            # Extract
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=CAPSULES_DIR)

            # Cleanup tar file
            os.remove(tar_path)
            return f"{capsule_id}: success"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                return f"{capsule_id}: FAILED - {e}"

    return f"{capsule_id}: FAILED after {max_retries} attempts"

def main():
    # Check if core_test.json exists
    if not os.path.exists(CORE_TEST_PATH):
        encrypted = os.path.join(COREBENCH_DIR, "core_test.json.gpg")
        print(f"ERROR: {CORE_TEST_PATH} not found!")
        print(f"Decrypt it first with:")
        print(f"  gpg --output {CORE_TEST_PATH} --decrypt {encrypted}")
        print(f'  Password: "reproducibility"')
        return

    # Load dataset
    with open(CORE_TEST_PATH, 'r') as f:
        dataset = json.load(f)

    # Get unique capsule IDs
    capsule_ids = list(set(task["capsule_id"] for task in dataset))
    print(f"Found {len(capsule_ids)} unique capsules to download")

    # Create capsules directory
    os.makedirs(CAPSULES_DIR, exist_ok=True)

    # Download with parallel workers
    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_and_extract, cid): cid for cid in capsule_ids}

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(capsule_ids)}] {result}")

            if "FAILED" in result:
                failed.append(futures[future])

    print(f"\nDone! {len(capsule_ids) - len(failed)}/{len(capsule_ids)} successful")
    if failed:
        print(f"Failed capsules: {failed}")

if __name__ == "__main__":
    main()
