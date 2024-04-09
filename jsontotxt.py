import json
from pathlib import Path

data_dir = Path("data/squad")


def parse_squad(filepath, outname, out_dir):
    with open(filepath, "r") as file:
        node = json.load(file)

    src = []
    tgt = []
    remove_internal_linebreak = lambda line: line.strip().replace("\n", " ")
    for data in node["data"]:
        for paragraph in data["paragraphs"]:
            context = remove_internal_linebreak(paragraph["context"]) + "\n"
            questions = [
                remove_internal_linebreak(qas["question"]) + "\n"
                for qas in paragraph["qas"]
                if not qas["is_impossible"]
            ]

            src.extend([context] * len(questions))
            tgt.extend(questions)

    with open(Path(out_dir) / f"{outname}.src", "w") as out:
        out.writelines(src)
    with open(Path(out_dir) / f"{outname}.tgt", "w") as out:
        out.writelines(tgt)


parse_squad(data_dir / "train-v2.0.json", "train", data_dir)
parse_squad(data_dir / "dev-v2.0.json", "dev", data_dir)
