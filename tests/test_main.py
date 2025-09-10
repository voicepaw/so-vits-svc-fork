import json
import os
from pathlib import Path
from unittest import SkipTest, TestCase

IS_CI = os.environ.get("GITHUB_ACTIONS", False)
IS_COLAB = os.getenv("COLAB_RELEASE_TAG", False)


class TestMain(TestCase):
    def test_import(self):
        import so_vits_svc_fork.cluster.train_cluster
        import so_vits_svc_fork.inference.main

        # import so_vits_svc_fork.modules.onnx._export
        import so_vits_svc_fork.preprocessing.preprocess_flist_config
        import so_vits_svc_fork.preprocessing.preprocess_hubert_f0
        import so_vits_svc_fork.preprocessing.preprocess_resample
        import so_vits_svc_fork.preprocessing.preprocess_split
        import so_vits_svc_fork.train  # noqa

    def test_infer(self):
        if IS_CI:
            raise SkipTest("Skip inference test on CI")
        from so_vits_svc_fork.inference.main import infer  # noqa

        # infer("tests/dataset_raw/34j/1.wav", "tests/configs/config.json", "tests/logs/44k")

    def test_preprocess(self):
        from so_vits_svc_fork.preprocessing.preprocess_resample import (
            preprocess_resample,
        )

        preprocess_resample("tests/dataset_raw", "tests/dataset/44k", 44100, n_jobs=1 if IS_CI else -1)

        from so_vits_svc_fork.preprocessing.preprocess_flist_config import (
            preprocess_config,
        )

        preprocess_config(
            "tests/dataset/44k",
            "tests/filelists/train.txt",
            "tests/filelists/val.txt",
            "tests/filelists/test.txt",
            "tests/configs/44k/config.json",
            "so-vits-svc-4.0v1",
        )

        if IS_CI:
            raise SkipTest("Skip hubert and f0 test on CI")
        from so_vits_svc_fork.preprocessing.preprocess_hubert_f0 import (
            preprocess_hubert_f0,
        )

        preprocess_hubert_f0("tests/dataset/44k", "tests/configs/44k/config.json")

    def test_train(self):
        if not IS_COLAB:
            raise SkipTest("Skip training test on non-colab")
        # requires >10GB of GPU memory, can be only tested on colab
        from so_vits_svc_fork.train import train

        config_path = Path("tests/logs/44k/config.json")
        config_json = json.loads(config_path.read_text("utf-8"))
        config_json["train"]["epochs"] = 1
        config_path.write_text(json.dumps(config_json), "utf-8")
        train(config_path, "tests/logs/44k")
