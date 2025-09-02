# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""The Open WebText Corpus (Zenodo shards only)."""

import re
from pathlib import Path

import datasets


_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
An open-source replication of the WebText dataset from OpenAI.
This loader uses only local Zenodo-style tar.xz shards that contain .txt files.
"""

# Compile once for small cleanup of excessive blank lines
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")


class Openwebtext(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset (Zenodo layout only)."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Only use local Zenodo-style shards: owt/openwebtext/*.xz
        repo_root = Path(__file__).resolve().parents[1]
        zenodo_dir = repo_root / "openwebtext"

        if not zenodo_dir.exists():
            raise FileNotFoundError(
                f"Expected Zenodo shards under {zenodo_dir}. Place your *.xz files there."
            )

        xz_files = sorted(str(p) for p in zenodo_dir.glob("*.xz"))
        if not xz_files:
            raise FileNotFoundError(f"No .xz shards found in {zenodo_dir}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "xz_files": xz_files,
                    "iter_archive": dl_manager.iter_archive,
                },
            ),
        ]

    def _generate_examples(self, xz_files, iter_archive):
        """Yields examples from Zenodo-style tar.xz shards containing .txt files."""
        for xz_path in xz_files:
            shard_name = Path(xz_path).name
            for inner_path, f in iter_archive(xz_path):
                if not inner_path.endswith(".txt"):
                    continue
                # Build a stable, shard-qualified ID to avoid collisions
                idx = f"{shard_name}/{inner_path}"
                yield idx, {"text": re.sub("\n\n\n+", "\n\n", f.read().decode("utf-8")).strip()}