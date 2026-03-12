import argparse
import tarfile


def extract_tar(source_path: str, target_path: str):
    """
    Extract the contents of a tar archive file to a specified directory.

    Args:
        source_path (str): Path to the tar archive file.
        target_path (str): Path to the folder where the tar contents will be extracted.
    """
    print(f"Extracting dataset to {target_path}...")
    with tarfile.open(source_path, "r:xz") as tar:
        tar.extractall(target_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path", type=str, default="mvtec_anomaly_detection.tar.xz",
        help="path to the dataset .tar.xz file"
    )

    args = parser.parse_args()

    extract_tar(args.dataset_path, "./data")
