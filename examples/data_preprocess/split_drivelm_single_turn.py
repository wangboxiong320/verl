import argparse
import json
from typing import List, Dict, Any


def replace_prefix(path: str, old_prefix: str, new_prefix: str) -> str:
    if old_prefix and path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix):]
    return path


def process_record(
    record: Dict[str, Any], old_prefix: str, new_prefix: str
) -> List[Dict[str, Any]]:
    images_key = "image" if "image" in record else ("images" if "images" in record else None)
    if images_key is None:
        return []

    images = record.get(images_key, [])
    images = [replace_prefix(p, old_prefix, new_prefix) for p in images]

    conv_key = "conversations" if "conversations" in record else ("conversation" if "conversation" in record else None)
    if conv_key is None:
        return []

    conversations = record.get(conv_key, [])

    pending_human = None
    outputs: List[Dict[str, Any]] = []

    for turn in conversations:
        role_raw = turn.get("from") or turn.get("role")
        role = "human" if role_raw in ("human", "user") else ("gpt" if role_raw in ("gpt", "assistant") else role_raw)
        value = turn.get("value") or turn.get("content") or ""

        if role == "human":
            pending_human = {"from": "human", "value": value}
        elif role == "gpt":
            if pending_human is not None:
                single_conv = [pending_human, {"from": "gpt", "value": value}]
                new_item = {
                    images_key: images,
                    conv_key: single_conv,
                }
                outputs.append(new_item)
                pending_human = None

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/drivelm_change_box_type.jsonl",help="输入JSONL文件路径")
    parser.add_argument("--output", default="./data/drivelm_single_turn.jsonl",help="输出JSONL文件路径")
    parser.add_argument("--old_prefix", default="/mnt/evad_fs/opensource-data", help="需要被替换的旧前缀")
    parser.add_argument("--new_prefix", default="/home/dataset-assist-0/wangboxiong/dataset", help="替换后的新前缀")
    args = parser.parse_args()

    current_id = 0
    written = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            records = process_record(data, args.old_prefix, args.new_prefix)
            for rec in records:
                out = {"id": current_id}
                out.update(rec)
                current_id += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} single-turn records to {args.output}")


if __name__ == "__main__":
    main()


