import tarfile

with tarfile.open("data/mvtec_anomaly_detection.tar.xz", "r:xz") as tar:
    tar.extractall("data/")
