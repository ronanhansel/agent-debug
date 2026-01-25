#!/usr/bin/env python3
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Recursively rename files and directories containing a specific Run ID.")
    parser.add_argument("--from", dest="from_id", required=True, help="The source Run ID/Timestamp substring to replace (e.g., 20260125_083839)")
    parser.add_argument("--to", dest="to_id", required=True, help="The target Run ID/Timestamp substring (e.g., 20260124_180412)")
    parser.add_argument("--scope", default=".hal_data", help="Root directory to search (default: .hal_data)")
    
    args = parser.parse_args()
    
    root_path = os.path.abspath(args.scope)
    old_id = args.from_id
    new_id = args.to_id
    
    if not os.path.exists(root_path):
        print(f"Error: Scope directory '{root_path}' does not exist.")
        sys.exit(1)
        
    print(f"Scanning '{root_path}' to rename '{old_id}' -> '{new_id}'...")
    print("----------------------------------------------------------------")
    
    count = 0
    
    # Walk bottom-up (topdown=False) is CRITICAL.
    # We must rename children (files/subdirs) before we rename the parent directory,
    # otherwise the paths to the children would become invalid before we reach them.
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        
        # 1. Rename files in the current directory
        for filename in filenames:
            if old_id in filename:
                old_file_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(old_id, new_id)
                new_file_path = os.path.join(dirpath, new_filename)
                
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"File: {filename} -> {new_filename}")
                    count += 1
                except Exception as e:
                    print(f"ERROR renaming file '{old_file_path}': {e}")

        # 2. Rename subdirectories in the current directory
        for dirname in dirnames:
            if old_id in dirname:
                old_dir_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(old_id, new_id)
                new_dir_path = os.path.join(dirpath, new_dirname)
                
                try:
                    os.rename(old_dir_path, new_dir_path)
                    print(f"Dir:  {dirname} -> {new_dirname}")
                    count += 1
                except Exception as e:
                    print(f"ERROR renaming dir '{old_dir_path}': {e}")

    print("----------------------------------------------------------------")
    print(f"Done. Renamed {count} items.")

if __name__ == "__main__":
    main()
