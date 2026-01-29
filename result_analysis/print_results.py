import json
import re

from google.cloud import storage

import matplotlib.pyplot as plt

TEST_PATTERN = "\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}[ \t]+-[ \t]*\w+[ \t]*-[ \t]*\w+[ \t]*-[ \t]+test_my_metric[ \t]+:[ \t]+(.*)"
TRAIN_PATTERN = "\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}[ \t]+-[ \t]*\w+[ \t]*-[ \t]*\w+[ \t]*-[ \t]+my_metric[ \t]+:[ \t]+(.*)"


def list_and_read_files_in_subdir(bucket_name, subdirectory):
    """Lists and reads files from a GCS subdirectory."""
    storage_client = storage.Client()

    # Note: The prefix should not start with a '/' and should end with one
    # to ensure you're matching a directory.
    if not subdirectory.endswith("/"):
        subdirectory += "/"

    # Use list_blobs to get all files in the subdirectory
    blobs = storage_client.list_blobs(bucket_name, prefix=subdirectory)

    print(f"Files in gs://{bucket_name}/{subdirectory}:")
    for blob in blobs:
        # blob.name includes the full path (e.g., 'dir1/subdir2/file.txt')
        # We can skip directories themselves if they appear as empty blobs
        if not blob.name.endswith(".log"):
            continue

        print(f"\n--- Opening: {blob.name} ---")

        try:
            # Download the contents of the blob as a string
            # Use .decode('utf-8') to convert bytes to a string
            content = blob.download_as_string().decode("utf-8")
            yield blob.name, content
        except Exception as e:
            print(f"Could not read file {blob.name}: {e}")


def parse_file(content, pattern):
    res = []
    lines = content.split("\n")
    for line in content.split("\n"):
        res += re.findall(pattern, line)
    res = [float(x) for x in res]
    return res


def load(bucket_name, subdirectory, pattern):
    result = {}
    offset = len(subdirectory.split("/"))
    for name, content in list_and_read_files_in_subdir(bucket_name, subdirectory):
        parsed = parse_file(content, pattern)
        name_list = name.split("/")
        res = result
        for n in name_list[offset:-2]:
            if n not in res:
                res[n] = {}
            res = res[n]
        res[name_list[-2]] = parsed

    return result


def main(bucket_name, subdirectory):
    for pattern in [TRAIN_PATTERN, TEST_PATTERN]:
        result = load(bucket_name, subdirectory, pattern)
        print(result.keys())
        # print(json.dumps(result, indent=2))
        for k, res in result.items():
            plt.figure(k)

            for x in res.values():
                if pattern == TRAIN_PATTERN:
                    plt.plot(x, "--")
                else:
                    plt.plot(x)

    plt.show()


if __name__ == "__main__":
    main("fr-rde-dest-smartlabel-dev-noisy-slack", "results")
