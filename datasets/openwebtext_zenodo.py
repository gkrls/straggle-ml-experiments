# coding=utf-8
import re
from pathlib import Path
import datasets

_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished={\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""

_DESCRIPTION = """\
OpenWebText from local Zenodo-style tar.xz shards containing .txt files.
"""

_RE_MANY_NEWLINES = re.compile(r"\n{3,}")

class Openwebtext(datasets.GeneratorBasedBuilder):
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
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        zenodo_dir = Path(__file__).resolve().parent / "openwebtext"

        if not zenodo_dir.exists():
            raise FileNotFoundError(f"Expected shards under {zenodo_dir}")

        xz_files = sorted(str(p) for p in zenodo_dir.glob("*.xz"))
        if not xz_files:
            raise FileNotFoundError(f"No .xz shards found in {zenodo_dir}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"xz_files": xz_files, "iter_archive": dl_manager.iter_archive},
            ),
        ]

    def _generate_examples(self, xz_files, iter_archive):
        for xz_path in xz_files:
            shard_name = Path(xz_path).name
            for inner_path, f in iter_archive(xz_path):
                if not inner_path.endswith(".txt"):
                    continue
                text = _RE_MANY_NEWLINES.sub("\n\n", f.read().decode("utf-8")).strip()
                yield f"{shard_name}/{inner_path}", {"text": text}
