#!/usr/bin/env python3
"""
実験⑬の長文脈版(提案手法)。scripts/finetune_ainu_watanabe_recipe.py(条件A、
発話単位)と同じHFベースのレシピ(渡辺さんのnotebook設定)に、lcasrの
sequence_scheduler(exp/train.py + lcasr/utils/scheduling.py)と同じ考え方の
「学習が進むにつれてチャンク長を伸ばすカリキュラム」をTrainerCallbackとして
追加したもの。

設計のポイント: HFの`Trainer`は`train_dataset`を一度だけ`DataLoader`化して
使い回すため、lcasr/exp/train.pyのように毎epochでdataloaderを再構築する
方式はそのままでは使えない。代わりに、`AinuLongContextCTCDataset.__len__`を
(カリキュラム開始時=最短チャンク長での)固定値にしておき、`__getitem__`が
呼ばれた瞬間に「今のチャンク長」を共有状態(SequenceWarmupState)から読んで
その場でチャンク境界を計算する設計にした。これにより`Trainer.train()`の
ループ自体は一切変更せずに、カリキュラムをCallbackだけで実現できる。
長文脈化に伴うバッチサイズ縮小(lcasrのbatch_size_multiplier)は、こちらは
既にper_device_batch_size=1(最小)のため不要。

評価の近似: 学習中のdev評価は「その時点のチャンク長で切ったチャンク単位」の
WERであり、コレクション全体を1つながりで評価するlcasrのfetch_logits方式の
dev評価とは厳密には異なる近似値。最終test評価は、学習完了時点の最大チャンク長
でコレクションを等分割し、チャンクごとに推論した予測文をコレクション内の順序で
連結してコレクション単位のWERを計算する(オーバーラップさせたlogitsのマージは
行わない、簡易版)。

Usage:
  uv run --extra gpu python scripts/finetune_ainu_watanabe_recipe_longcontext.py \\
    --data-dir data/ainu_watanabe_longcontext_kfold/fold0 \\
    --vocab-path lcasr/artifacts/ainu_hftok_kfold/fold0/vocab.json \\
    --output-dir checkpoints/ainu_watanabe_longcontext_kfold/fold0
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from jiwer import wer, cer
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

SR = 16000


def load_collections(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


class SequenceWarmupState:
    """train_datasetとcallbackの間で共有する、現在のチャンク長(秒)。"""

    def __init__(self, start_length_sec: float):
        self.cur_length_sec = start_length_sec


class SequenceWarmupCallback(TrainerCallback):
    """lcasr/utils/scheduling.SequenceWarmupManagerと同じスケジュール
    (increase_every stepごとにincrease_by_multiplier倍、max_length_secで止める)。"""

    def __init__(self, state: SequenceWarmupState, increase_every: int,
                 increase_by_multiplier: float, max_length_sec: float):
        self.state = state
        self.increase_every = increase_every
        self.increase_by_multiplier = increase_by_multiplier
        self.max_length_sec = max_length_sec

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.increase_every == 0:
            new_len = self.state.cur_length_sec * self.increase_by_multiplier
            self.state.cur_length_sec = min(new_len, self.max_length_sec)


class AinuLongContextCTCDataset(Dataset):
    """コレクション単位で連結した生波形を、現在のチャンク長(warmup_state)で
    等分割して1サンプル=1チャンクとして返す。__len__は固定値(num_slots)で、
    __getitem__が呼ばれるたびにその時点のチャンク長を反映したチャンク一覧
    (メモ化)の中から idx % 現在のチャンク数 を返す。"""

    def __init__(self, collections: List[dict], processor: Wav2Vec2Processor,
                 warmup_state: SequenceWarmupState, num_slots: int):
        self.collections = collections
        self.processor = processor
        self.warmup_state = warmup_state
        self.num_slots = num_slots
        self._chunk_cache_key = None
        self._chunk_cache: List[tuple] = []
        self._words_cache: Dict[int, list] = {}

    def __len__(self):
        return self.num_slots

    def _words_for(self, coll_idx: int) -> list:
        if coll_idx not in self._words_cache:
            self._words_cache[coll_idx] = json.loads(
                Path(self.collections[coll_idx]['txt']).read_text(encoding='utf-8'))
        return self._words_cache[coll_idx]

    def _chunks_for_current_length(self):
        key = round(self.warmup_state.cur_length_sec, 2)
        if key == self._chunk_cache_key:
            return self._chunk_cache
        length = self.warmup_state.cur_length_sec
        chunks = []
        for coll_idx, rec in enumerate(self.collections):
            duration = rec['duration']
            n_chunks = max(1, math.ceil(duration / length))
            for c in range(n_chunks):
                start = c * length
                end = min(start + length, duration)
                if end - start < 1.0 and c > 0:
                    continue  # 末尾の1秒未満の余りはスキップ(前チャンクに含まれる範囲とほぼ同じ)
                chunks.append((coll_idx, start, end))
        self._chunk_cache_key = key
        self._chunk_cache = chunks
        return chunks

    def __getitem__(self, idx):
        chunks = self._chunks_for_current_length()
        coll_idx, start, end = chunks[idx % len(chunks)]
        rec = self.collections[coll_idx]

        full_waveform = torch.load(rec['audio'], map_location='cpu', weights_only=True)
        start_sample, end_sample = int(start * SR), int(end * SR)
        waveform = full_waveform[..., start_sample:end_sample].squeeze(0).numpy()

        words = self._words_for(coll_idx)
        chunk_words = [w['text'] for w in words if w['start'] >= start and w['end'] <= end]
        text = ' '.join(chunk_words)

        input_values = self.processor(waveform, sampling_rate=SR).input_values[0]
        labels = self.processor.tokenizer(text).input_ids
        return {'input_values': input_values, 'labels': labels, 'length': len(input_values)}


@dataclass
class DataCollatorCTCWithPadding:
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


def decode_collection(model, processor, device, rec: dict, chunk_length_sec: float) -> str:
    """最大チャンク長でコレクションを等分割し、チャンクごとの予測文を連結する
    (オーバーラップlogitsのマージは行わない簡易評価、docstring参照)。"""
    full_waveform = torch.load(rec['audio'], map_location='cpu', weights_only=True)
    duration = rec['duration']
    n_chunks = max(1, math.ceil(duration / chunk_length_sec))
    pieces = []
    for c in range(n_chunks):
        start, end = c * chunk_length_sec, min((c + 1) * chunk_length_sec, duration)
        start_sample, end_sample = int(start * SR), int(end * SR)
        waveform = full_waveform[..., start_sample:end_sample].squeeze(0).numpy()
        if waveform.shape[-1] < SR * 0.1:
            continue
        input_values = processor(waveform, sampling_rate=SR).input_values[0]
        with torch.no_grad():
            logits = model(torch.tensor(input_values, device=device).unsqueeze(0)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pieces.append(processor.batch_decode(pred_ids.cpu().numpy())[0])
    return ' '.join(p for p in pieces if p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help='train/dev/test_collections.json(scripts/prepare_ainu_wav2vec2_longcontext_raw.py)')
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-model', default='karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=4500)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--logging-steps', type=int, default=20)
    # lcasrの実験③/⑤条件Bと同じ値(audio_chunking.size=256frames@50Hz=5.12s,
    # max_sequence_length=8192frames=163.84s、increase_every=200,倍率2)。
    parser.add_argument('--start-length-sec', type=float, default=5.12)
    parser.add_argument('--max-length-sec', type=float, default=163.84)
    parser.add_argument('--increase-every', type=int, default=200)
    parser.add_argument('--increase-by-multiplier', type=float, default=2.0)
    parser.add_argument('--num-slots', type=int, default=None,
                        help='Dataset.__len__の固定値。省略時は開始チャンク長での'
                             'チャンク総数(最も細かく分割した時の数)から自動算出')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Wav2Vec2CTCTokenizer(
        args.vocab_path, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SR, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(output_dir))

    train_collections = load_collections(data_dir / 'train_collections.json')
    dev_collections = load_collections(data_dir / 'dev_collections.json')
    test_collections = load_collections(data_dir / 'test_collections.json')
    print(f'train: {len(train_collections)} collections  '
          f'dev: {len(dev_collections)}  test: {len(test_collections)}')

    warmup_state = SequenceWarmupState(args.start_length_sec)

    if args.num_slots is None:
        total_duration = sum(r['duration'] for r in train_collections)
        num_slots = max(1, math.ceil(total_duration / args.start_length_sec))
    else:
        num_slots = args.num_slots
    print(f'num_slots (fixed dataset length): {num_slots}')

    train_dataset = AinuLongContextCTCDataset(train_collections, processor, warmup_state, num_slots)
    # devも同じ現在チャンク長で評価する(近似、docstring参照)。
    dev_slots = max(1, math.ceil(sum(r['duration'] for r in dev_collections) / args.start_length_sec))
    dev_dataset = AinuLongContextCTCDataset(dev_collections, processor, warmup_state, dev_slots)

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
        return {'wer': wer(label_str, pred_str), 'cer': cer(label_str, pred_str),
                'chunk_length_sec': warmup_state.cur_length_sec}

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        report_to=['tensorboard'],
        dataloader_num_workers=2,
    )

    warmup_callback = SequenceWarmupCallback(
        warmup_state, args.increase_every, args.increase_by_multiplier, args.max_length_sec)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[warmup_callback],
    )

    existing_ckpts = sorted(output_dir.glob('checkpoint-*'))
    if existing_ckpts:
        print(f'resuming from {existing_ckpts[-1]}')
    trainer.train(resume_from_checkpoint=bool(existing_ckpts))
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f'Saved best model to {output_dir}, final chunk_length_sec={warmup_state.cur_length_sec}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    results = []
    for rec in test_collections:
        hyp = decode_collection(model, processor, device, rec, args.max_length_sec)
        ref = ' '.join(w['text'] for w in json.loads(Path(rec['txt']).read_text(encoding='utf-8')))
        results.append({'id': rec['id'], 'reference': ref, 'prediction': hyp})

    refs = [r['reference'] for r in results]
    hyps = [r['prediction'] for r in results]
    test_wer, test_cer = wer(refs, hyps), cer(refs, hyps)
    print(f'test metrics: wer={test_wer:.4f} cer={test_cer:.4f}')
    with open(output_dir / 'test_predictions.json', 'w', encoding='utf-8') as f:
        json.dump({'test_wer': test_wer, 'test_cer': test_cer, 'results': results}, f,
                   ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
