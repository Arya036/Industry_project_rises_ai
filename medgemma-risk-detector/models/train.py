"""
models/train.py

MedGemma / VLM Fine-Tuning Script using QLoRA (4-bit NF4 + LoRA).
Designed to run on a CUDA-enabled GPU (e.g. Kaggle T4 or local server).

Usage:
    python models/train.py --data_dir data/ --output_dir models/
"""

import os
import re
import argparse
import torch
import gc
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

# ── User Prompt Template ──────────────────────────────────────
USER_PROMPT_TEMPLATE = """You are an expert radiologist specializing in occupational lung disease.
Analyze this chest X-ray from a mining worker.

Computer-aided detection findings:
{findings_text}

Lung zone annotations:
{zones_text}

Generate a structured clinical report with:
EXAMINATION, FINDINGS (including zones affected), and Impression."""


class MedGemmaFineTuneDataset(Dataset):
    """Dataset class for MedGemma report generation training."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image": Image.open(s["image_path"]).convert("RGB").resize((336, 336)),
            "report_text": s["report_text"],
            "user_prompt": s["user_prompt"],
        }


def make_collate_fn(processor):
    START_OF_TURN = 105   # <start_of_turn> token
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(batch):
        all_inputs = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text",  "text" : item["user_prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["report_text"]}],
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=768,
            )
            all_inputs.append(inputs)

        # Pad across batch
        seq_keys = {"input_ids", "attention_mask", "token_type_ids"}
        max_seq_len = max(x["input_ids"].shape[-1] for x in all_inputs)

        pad_values = {
            "input_ids": pad_id,
            "attention_mask": 0,
            "token_type_ids": 0,
        }

        result = {}
        for key in all_inputs[0].keys():
            tensors = []
            for x in all_inputs:
                t = x[key]
                if key in seq_keys:
                    pad_len = max_seq_len - t.shape[-1]
                    if pad_len > 0:
                        padding = torch.full((1, pad_len), pad_values[key], dtype=t.dtype)
                        t = torch.cat([t, padding], dim=-1)
                tensors.append(t)
            result[key] = torch.cat(tensors, dim=0)

        # Labels - mask prompt tokens to avoid calculating loss on prompts
        labels = result["input_ids"].clone()
        labels[labels == pad_id] = -100

        for i in range(labels.shape[0]):
            ids = result["input_ids"][i]
            seq_len = ids.shape[0]

            last_sot = -1
            for j in range(seq_len):
                if ids[j].item() == START_OF_TURN:
                    last_sot = j

            if last_sot != -1:
                # Mask <start_of_turn> + "model" + newline
                labels[i, : last_sot + 3] = -100
            else:
                labels[i, : seq_len // 2] = -100

        result["labels"] = labels
        return result

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="MedGemma QLoRA Fine-tuning")
    parser.add_argument("--base_model", type=str, default="google/medgemma-4b-it", help="HF model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save adapter config and weights")
    args = parser.parse_args()

    print("Verifying GPU availability...")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Fine-tuning requires a GPU.")
        # We exit gracefully or proceed with CPU just for testing
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"VRAM Free: {free_gb:.1f} GB")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    print(f"Loading processor and base model: {args.base_model}")
    try:
        processor = AutoProcessor.from_pretrained(args.base_model)
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map={"default": "cuda:0"} if torch.cuda.is_available() else None,
        )
    except Exception as e:
        print(f"Failed to load model. Note: this script requires HF login & base model download access: {e}")
        return

    # PEFT LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    if torch.cuda.is_available():
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    model.print_trainable_parameters()

    # Note: Dummy list of samples just to demonstrate training pipeline setup.
    # In practice, attach your own clinical data directory.
    print("Preparing training datasets...")
    samples = []
    
    # Check if we have sample files in data directory to dry-run
    # For actual training, load CSV containing patient_id, image_path, report_text, findings
    # This is a template list of training samples
    dummy_sample = {
        "patient_id": "normal_001",
        "image_path": "data/normal_001.jpg",
        "user_prompt": USER_PROMPT_TEMPLATE.format(
            findings_text="- Multiple Nodules: Absent\n- Hilum Abnormality: Absent",
            zones_text="- No spatial annotations available"
        ),
        "report_text": "EXAMINATION: CHEST (PA)\nFINDINGS: Lungs are clear. No focal consolidation or effusion.\nImpression: Normal chest study."
    }
    
    if os.path.exists(dummy_sample["image_path"]):
        samples = [dummy_sample] * 10
    else:
        print("Data files not found. Setup dummy data list for compilation verification.")
        samples = []

    if len(samples) == 0:
        print("No training data found. Exiting train dry-run.")
        return

    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    train_ds = MedGemmaFineTuneDataset(train_samples)
    val_ds = MedGemmaFineTuneDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=0
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=max(total_steps, 1)
    )

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting fine-tuning loop: {args.epochs} epochs, batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            total_loss += outputs.loss.detach().item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 5 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)} | Avg Loss: {avg:.4f}")

        # Eval
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss_total += outputs.loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss_total / max(len(val_loader), 1)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  ✓ Saved best checkpoint to {args.output_dir} (Val Loss: {avg_val_loss:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training process finished.")


if __name__ == "__main__":
    main()
