# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DriveLM dataset to parquet format
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def split_train_test(data: List[Dict[str, Any]], test_ratio: float = 0.1, random_seed: int = 42) -> tuple:
    """Split data into train and test sets"""
    random.seed(random_seed)
    random.shuffle(data)
    
    test_size = int(len(data) * test_ratio)
    test_data = data[:test_size]
    train_data = data[test_size:]
    
    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./data/drivelm_single_turn.jsonl", help="Path to input JSONL file")
    parser.add_argument("--local_dir", default="../../data/drivelm", help="Local output directory")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS output directory")
    parser.add_argument("--test_ratio", default=0.1, type=float, help="Ratio of test data")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for splitting")

    args = parser.parse_args()

    # Load data from JSONL file
    print(f"Loading data from {args.input_file}")
    all_data = load_jsonl_data(args.input_file)
    print(f"Loaded {len(all_data)} samples")
    
    # Split into train and test
    train_data, test_data = split_train_test(all_data, args.test_ratio, args.random_seed)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Instruction for multi-modal autonomous driving tasks
    instruction_following = (
        r"You are an autonomous driving assistant analyzing multi-view camera images from an ego vehicle. "
        r"The images show different perspectives: FRONT VIEW, FRONT LEFT VIEW, FRONT RIGHT VIEW, BACK LEFT VIEW, BACK RIGHT VIEW, and BACK VIEW. "
        r"When you see bounding box coordinates in the format <VIEW><box>[x1,y1,x2,y2]</box>, these represent normalized coordinates "
        r"where [x1,y1] is the top-left corner and [x2,y2] is the bottom-right corner of the bounding box, "
        r"with all coordinates normalized to the range [0, 1000]. "
        r"Please analyze the driving scene, identify objects, their status, and provide appropriate driving decisions. "
        # r"Your reasoning process should be enclosed within <think> </think> tags."
    )

    def process_drivelm_sample(sample: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
        """Process a single-turn DriveLM sample to unified parquet schema (geo3k-style)."""
        sample_id = sample.get("id")
        images_field = "image" if "image" in sample else ("images" if "images" in sample else None)
        images = sample.get(images_field, []) if images_field else []
        conv_field = "conversations" if "conversations" in sample else ("conversation" if "conversation" in sample else None)
        conversations = sample.get(conv_field, []) if conv_field else []

        # Expect single-turn: [human(question), gpt(answer)]
        question = ""
        answer = ""
        if len(conversations) >= 1:
            first = conversations[0]
            if (first.get("from") == "human") or (first.get("role") == "user"):
                question = first.get("value") or first.get("content") or ""
        if len(conversations) >= 2:
            second = conversations[1]
            if (second.get("from") == "gpt") or (second.get("role") == "assistant"):
                answer = second.get("value") or second.get("content") or ""

        # Build single message prompt like geo3k
        prompt = question + " " + instruction_following
        
        processed_data = {
            "data_source": "drivelm",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": images,
            "ability": "autonomous_driving",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "sample_id": sample_id,
                "conversation_length": len(conversations),
                "question": question,
                "answer": answer,
            },
        }
        return processed_data

    # Process train and test data
    print("Processing training data...")
    train_processed = [process_drivelm_sample(sample, "train", idx) for idx, sample in enumerate(train_data)]
    
    print("Processing test data...")
    test_processed = [process_drivelm_sample(sample, "test", idx) for idx, sample in enumerate(test_data)]

    # Create output directory
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    # Convert to datasets format and save
    print("Converting to datasets format and saving...")
    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    print(f"Saved train dataset with {len(train_dataset)} samples to {os.path.join(local_dir, 'train.parquet')}")
    print(f"Saved test dataset with {len(test_dataset)} samples to {os.path.join(local_dir, 'test.parquet')}")

    # Upload to HDFS if specified
    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        print(f"Uploading to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("Upload completed!")
