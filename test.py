import lmdb
import os


def merge_lmdb(lmdb_paths, output_lmdb_path):
    # Create the output LMDB environment
    if not os.path.exists(output_lmdb_path):
        os.makedirs(output_lmdb_path)

    # Set a larger map size if your data is large
    map_size = 10 * 1024 * 1024 * 1024  # 10 GB, adjust if necessary

    # Open the output LMDB
    output_env = lmdb.open(output_lmdb_path, map_size=map_size)

    # Iterate over each LMDB file
    for db_index, db_path in enumerate(lmdb_paths):
        print(f"Merging {db_path} into {output_lmdb_path} with prefix {db_index}_")

        # Open the input LMDB
        with lmdb.open(db_path, readonly=True) as input_env:
            with input_env.begin() as input_txn:
                with output_env.begin(write=True) as output_txn:
                    # Iterate over all key-value pairs in the input LMDB
                    for key, value in input_txn.cursor():
                        # Prefix the key to ensure uniqueness
                        unique_key = f"{db_index}_{key.decode('utf-8')}".encode("utf-8")
                        # Put the modified key and original value into the output LMDB
                        output_txn.put(unique_key, value)

    # Close the output LMDB
    output_env.close()
    print("Merging complete.")


# Example usage:
lmdb_files = ["path/to/lmdb1", "path/to/lmdb2", "path/to/lmdb3"]
output_lmdb = "path/to/output_lmdb"
merge_lmdb(lmdb_files, output_lmdb)
