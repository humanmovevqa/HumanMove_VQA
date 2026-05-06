# LLamaFactory — SFT training config

Config for supervised fine-tuning of **Qwen3-VL 8B** on the spatial motion QA data produced by the MotionScript pipeline. Uses [hiyouga/LLamaFactory](https://github.com/hiyouga/LlamaFactory) (ACL 2024).

---

## Setup

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

---

## Registering the dataset

LLamaFactory requires a `dataset_info.json` in `dataset_dir` that registers each split. Add entries for your train and eval JSONL files produced by `prepare_sft_data.py`:

```json
{
  "my_train": {
    "file_name": "train_answer_only.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "videos": "video" }
  },
  "my_eval": {
    "file_name": "test_answer_only.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "videos": "video" }
  }
}
```

Place this file alongside the JSONL files and set `dataset_dir` in `sft_answer.yaml` to that directory.

---

## Config: `sft_answer.yaml`

| Section | Key settings |
|---|---|
| **Model** | `Qwen/Qwen3-VL-8B-Instruct`, `video_maxlen=64`, `video_fps=64` (uniform sampling) |
| **Method** | LoRA, rank 16, alpha 32, dropout 0.05, all modules targeted |
| **Data** | Set `dataset_dir`, `dataset`, `eval_dataset` to your registered names |
| **Output** | Set `output_dir` for checkpoints and TensorBoard logs |
| **Training** | 5 epochs, LR 5e-5, cosine schedule, warmup 10%, bf16, batch 2 × grad-accum 2 |
| **Eval** | Every 500 steps |

The `default_system` prompt instructs the model on the world coordinate convention used by the spatial caption pipeline (Y-up, motion relative to first frame, quarter-based temporal references).

---

## Running

Fill in `dataset_dir` and `output_dir` in `sft_answer.yaml`, then:

```bash
llamafactory-cli train scripts/llamafactory/sft_answer.yaml
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir /path/to/output/saves/
```

---

## Notes

- The `template: qwen3_vl_nothink` disables Qwen3's built-in chain-of-thought token, keeping assistant responses in the plain `Answer: X` format expected by the eval pipeline.
- To train on CoT data (`train_cot.jsonl`), register a separate dataset entry pointing to that file; the `evidence` field is ignored by LLamaFactory's sharegpt formatter.
- For multi-GPU training add `--nproc_per_node <N>` via `torchrun` or configure DeepSpeed with `pip install -r requirements/deepspeed.txt`.
