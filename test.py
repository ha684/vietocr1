import lmdb
import struct


def merge_lmdb(source_paths, target_path):
    with lmdb.open(target_path, map_size=10 * 1024 * 1024 * 1024) as target_env:
        with target_env.begin(write=True) as target_txn:
            total_entries = 0
            for source_index, source_path in enumerate(source_paths):
                entries_in_source = 0
                with lmdb.open(source_path, readonly=True) as source_env:
                    with source_env.begin() as source_txn:
                        cursor = source_txn.cursor()
                        for key, value in cursor:
                            # Always append a unique identifier to the key
                            new_key = (
                                key
                                + b"__"
                                + struct.pack(">QQ", source_index, entries_in_source)
                            )
                            target_txn.put(new_key, value)
                            entries_in_source += 1
                total_entries += entries_in_source
                print(
                    f"Source {source_index + 1}: Added {entries_in_source} entries. Total entries: {total_entries}"
                )

    # Verify final count
    with lmdb.open(target_path, readonly=True) as env:
        with env.begin() as txn:
            final_count = txn.stat()["entries"]
    print(f"Final entry count in merged LMDB: {final_count}")


# Usage
source_paths = [
    r"D:\Workspace\python_code\vietocr1\train_ha",
    r"D:\Workspace\python_code\vietocr1\train_ha1",
]
target_path = "merged_lmdb"
merge_lmdb(source_paths, target_path)
