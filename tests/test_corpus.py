import subprocess, hashlib, os, shutil, json, sys, pathlib


def read(p):
    return open(p, "r", encoding="utf-8").read().strip()


def test_corpus_reproducible(tmp_path):
    # Re-ejecuta gen_corpus y compara hash con out/corpus_sha256.txt
    sha_saved = read("out/corpus_sha256.txt")
    out = subprocess.check_output(
        ["./tools/gen_corpus.sh", "42", "1a2b3c4d5e6f7890abcdef1234567890"]
    )
    sha_new = hashlib.sha256(out).hexdigest()
    assert sha_new == sha_saved
