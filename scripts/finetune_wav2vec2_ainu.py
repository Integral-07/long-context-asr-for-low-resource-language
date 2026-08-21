#!/usr/bin/env python3
"""
実験1: karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain を
Ainuコーパス(発話単位、短い音声、従来手法)でCTCファインチューニングする。

このリポジトリ(lcasr)のコードは使わない、HuggingFace transformers
のみによる独立したパイプライン。scripts/prepare_ainu_wav2vec2.py が
書き出した data/ainu_wav2vec2/{train,dev,test}.json + vocab.json を使う。

Usage:
  uv run --extra gpu python scripts/finetune_wav2vec2_ainu.py \\
    --data-dir data/ainu_wav2vec2 \\
    --output-dir ainu_wav2vec2_ft \\
    --base-model karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torchaudio
from jiwer import wer
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

SR = 16000


class AinuCTCDataset(Dataset):
    def __init__(self, records: List[dict], processor: Wav2Vec2Processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        waveform, sr = torchaudio.load(rec['audio'])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SR:
            waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
        waveform = waveform.squeeze(0).numpy()

        input_values = self.processor(
            waveform, sampling_rate=SR).input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(rec['text']).input_ids

        return {'input_values': input_values, 'labels': labels,
                'length': len(input_values)}


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': f['input_values']} for f in features]
        label_features = [{'input_ids': f['labels']} for f in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors='pt')
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=True, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch['labels'] = labels
        return batch


def load_records(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data/ainu_wav2vec2')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-model', default='karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain')
    parser.add_argument('--num-epochs', type=float, default=30)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--per-device-batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--eval-steps', type=int, default=200)
    parser.add_argument('--save-steps', type=int, default=200)
    parser.add_argument('--freeze-feature-encoder', action='store_true', default=True)
    parser.add_argument('--max-train-records', type=int, default=None,
                        help='スモークテスト用に学習件数を制限')
    parser.add_argument('--max-eval-records', type=int, default=None,
                        help='スモークテスト用に評価件数を制限')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = data_dir / 'vocab.json'
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SR, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(output_dir))

    train_records = load_records(data_dir / 'train.json')
    dev_records = load_records(data_dir / 'dev.json')
    if args.max_train_records:
        train_records = train_records[:args.max_train_records]
    if args.max_eval_records:
        dev_records = dev_records[:args.max_eval_records]
    print(f'train: {len(train_records)}  dev: {len(dev_records)}')

    train_dataset = AinuCTCDataset(train_records, processor)
    dev_dataset = AinuCTCDataset(dev_records, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction='mean',
        ctc_zero_infinity=True,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        return {'wer': wer(label_str, pred_str)}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        train_sampling_strategy='group_by_length',
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_batch_size,
        eval_strategy='steps',
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(output_dir / 'best'))
    processor.save_pretrained(str(output_dir / 'best'))
    print(f'Saved best model to {output_dir / "best"}')


if __name__ == '__main__':
    main()
