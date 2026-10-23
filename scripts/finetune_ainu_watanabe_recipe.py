#!/usr/bin/env python3
"""
実験⑬: 渡辺さんのfinetune_xlsr53_ainu notebook(ainu_finetune_xlsr53_ain.ipynb)の
セル8・10・11・13・15・17・19をほぼそのまま移植したスクリプト。

notebookとの違いは1点だけ: notebook自身はランダムなtrain/dev分割(テストセットなし)
を使うが、実験①(31.47%)など他の全実験と比較可能にするため、データ構築部分だけ
既存の固定split(data/ainu_wav2vec2/{train,dev,test}.json、
scripts/prepare_ainu_wav2vec2.pyが生成、held-out: at08,at30,at40,at13,at22)を使う。
トークナイザ構築・モデル設定・TrainingArgumentsはnotebookの値をそのまま使用。

Usage:
  uv run --extra gpu python scripts/finetune_ainu_watanabe_recipe.py \\
    --data-dir data/ainu_wav2vec2 \\
    --output-dir checkpoints/ainu_watanabe_recipe
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torchaudio
from jiwer import wer, cer
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

        input_values = self.processor(waveform, sampling_rate=SR).input_values[0]
        labels = self.processor.tokenizer(rec['text']).input_ids
        return {'input_values': input_values, 'labels': labels, 'length': len(input_values)}


@dataclass
class DataCollatorCTCWithPadding:
    """notebook cell 15と同一。"""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': f['input_values']} for f in features]
        label_features = [{'input_ids': f['labels']} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors='pt')
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels'] = labels
        return batch


def load_records(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


def filter_by_duration(records: List[dict], max_duration_seconds: float):
    kept = []
    for rec in records:
        info = torchaudio.info(rec['audio'])
        duration = info.num_frames / info.sample_rate
        if duration <= max_duration_seconds:
            kept.append(rec)
    return kept, len(records) - len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data/ainu_wav2vec2',
                        help='train/dev/test.json(固定split、notebook独自のランダム分割は使わない)')
    parser.add_argument('--vocab-path', default=None,
                        help='省略時は--data-dir/vocab.json。k-fold実行時はfold別トークナイザ'
                             '(scripts/train_ainu_hf_tokenizer.py出力、vocab leakage防止のため'
                             'データ準備とは別ディレクトリ)を指す')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-model', default='karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain')
    # 以下、notebook cell 19のTrainingArgumentsと同一の値。
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--per-device-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--max-duration-seconds', type=float, default=180.0,
                        help='train/dev用。notebook cell 13のMAX_INPUT_LEN(180秒)相当')
    parser.add_argument('--test-max-duration-seconds', type=float, default=30.0,
                        help='最終test評価専用のフィルタ。実験①(31.47%)・実験②③と同じ'
                             '「30秒以下209件」に揃えるためtrain/devとは別に指定する'
                             '(RESULT.mdの「限界」節に記載の通り、219件と209件は別の数値になる)')
    parser.add_argument('--max-train-records', type=int, default=None, help='スモークテスト用')
    parser.add_argument('--max-eval-records', type=int, default=None, help='スモークテスト用')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # notebook cell 11
    vocab_path = Path(args.vocab_path) if args.vocab_path else data_dir / 'vocab.json'
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SR, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(output_dir))

    train_records = load_records(data_dir / 'train.json')
    dev_records = load_records(data_dir / 'dev.json')
    test_records = load_records(data_dir / 'test.json')

    if args.max_duration_seconds:
        train_records, n_dropped_train = filter_by_duration(train_records, args.max_duration_seconds)
        dev_records, n_dropped_dev = filter_by_duration(dev_records, args.max_duration_seconds)
        print(f'duration filter (<= {args.max_duration_seconds}s): '
              f'dropped {n_dropped_train} train, {n_dropped_dev} dev')
    if args.test_max_duration_seconds:
        test_records, n_dropped_test = filter_by_duration(test_records, args.test_max_duration_seconds)
        print(f'test duration filter (<= {args.test_max_duration_seconds}s): dropped {n_dropped_test} test')
    if args.max_train_records:
        train_records = train_records[:args.max_train_records]
    if args.max_eval_records:
        dev_records = dev_records[:args.max_eval_records]
    print(f'train: {len(train_records)}  dev: {len(dev_records)}  test: {len(test_records)}')

    train_dataset = AinuCTCDataset(train_records, processor)
    dev_dataset = AinuCTCDataset(dev_records, processor)

    # notebook cell 17
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction='mean',
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return {'wer': wer(label_str, pred_str), 'cer': cer(label_str, pred_str)}

    # notebook cell 19
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        # notebookのgroup_by_length=Trueに相当(transformers 5.x API、
        # scripts/finetune_wav2vec2_ainu.pyで確認済みの対応)
        train_sampling_strategy='group_by_length',
        length_column_name='length',
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,
        # notebook cell 19と同じ。export_tensorboard_logs.py等の後処理は不要で、
        # Trainerが直接 <output_dir>/runs/ 以下にイベントファイルを書き出す。
        report_to=['tensorboard'],
        logging_dir=str(output_dir / 'runs'),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    existing_ckpts = sorted(output_dir.glob('checkpoint-*'))
    if existing_ckpts:
        print(f'resuming from {existing_ckpts[-1]}')

    trainer.train(resume_from_checkpoint=bool(existing_ckpts))
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f'Saved best model to {output_dir}')

    # 既存の固定テストセットで最終評価(他実験と比較可能な数値を出す)。
    # trainer.predict()を使い、集計値だけでなく発話単位の予測文も保存する
    # (eval_ainu_lcasr.pyのresults形式と揃え、k-fold実行時にfold間でpoolできるようにする)。
    test_dataset = AinuCTCDataset(test_records, processor)
    output = trainer.predict(test_dataset, metric_key_prefix='test')
    print('test metrics:', output.metrics)

    pred_ids = np.argmax(output.predictions, axis=-1)
    label_ids = output.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_strs = processor.batch_decode(pred_ids)
    label_strs = processor.batch_decode(label_ids, group_tokens=False)

    results = [
        {'id': rec['id'], 'reference': ref, 'prediction': pred}
        for rec, ref, pred in zip(test_records, label_strs, pred_strs)
    ]
    with open(output_dir / 'test_predictions.json', 'w', encoding='utf-8') as f:
        json.dump({**output.metrics, 'results': results}, f, ensure_ascii=False, indent=2)
    print(f'Saved test predictions to {output_dir / "test_predictions.json"}')


if __name__ == '__main__':
    main()
