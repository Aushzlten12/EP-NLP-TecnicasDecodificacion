import json, os, subprocess, pathlib


def test_make_targets_end_to_end():
    # Asegura que los targets principales generan artefactos esperados
    subprocess.check_call(["make", "data"])
    assert os.path.exists("out/corpus.txt")
    assert os.path.exists("out/corpus_sha256.txt")

    subprocess.check_call(["make", "tokenize"])
    assert os.path.exists("out/tokens.jsonl")
    assert os.path.exists("out/vocab.txt")

    subprocess.check_call(["make", "train"])
    assert os.path.exists("dist/model.tar.gz")

    subprocess.check_call(["make", "eval"])
    assert os.path.exists("out/metrics.json")
    with open("out/metrics.json", "r", encoding="utf-8") as f:
        m = json.load(f)
        assert "perplexity" in m


def test_pack_and_verify():
    subprocess.check_call(["make", "pack"])
    assert os.path.exists("dist/proy-v1.0.0.tar.gz")
    subprocess.check_call(["make", "verify"])
