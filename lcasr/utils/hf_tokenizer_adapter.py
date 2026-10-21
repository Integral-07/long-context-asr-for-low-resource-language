"""
実験⑫: 渡辺さんのfinetune_xlsr53_ainu notebookが使うHF `Wav2Vec2CTCTokenizer`を、
このリポジトリ内でSentencePieceProcessor(spm)が使われている箇所にそのまま差し込める
ようにするアダプタ。

lcasr全体はspmの `encode`/`decode`/`id_to_piece`/`get_piece_size`/`pad_id`/`vocab_size`
という狭いインターフェースにしか依存していない(exp/train.py, lcasr/utils/training.py,
lcasr/eval/utils.py, scripts/eval_ainu_lcasr.py で使用箇所を確認済み)ため、このクラスが
同じメソッド名を実装すれば他のコードは一切変更不要。

注意: HF `Wav2Vec2CTCTokenizer.decode()` はデフォルトで `group_tokens=True` となっており、
これは「生のモデル出力(blank含む)を自分でCTC折り畳みする」ためのオプション。lcasrでは
CTCの重複/blank折り畳みは常に呼び出し側(lcasr/decoding/greedy.pyのtorch.unique_consecutive等)
が先に済ませてから`decode()`を呼ぶので、ここで重ねてgroup_tokensを効かせると、"ee"のような
正当な連続同一文字まで1文字に潰してしまう。spmのdecode()と同じ「id列をそのまま文字列に戻す
だけ」の挙動に揃えるため、常に `group_tokens=False` で呼ぶ。
"""
from transformers import Wav2Vec2CTCTokenizer


class HFTokenizerAdapter:
    def __init__(self, hf_tokenizer: Wav2Vec2CTCTokenizer):
        self.hf = hf_tokenizer
        self._vocab_size = len(hf_tokenizer)

    @classmethod
    def from_pretrained(cls, path: str) -> 'HFTokenizerAdapter':
        return cls(Wav2Vec2CTCTokenizer.from_pretrained(path))

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids, group_tokens=False, skip_special_tokens=False)

    def id_to_piece(self, i: int) -> str:
        return self.hf.convert_ids_to_tokens(int(i))

    def get_piece_size(self) -> int:
        return self._vocab_size

    def vocab_size(self) -> int:
        return self._vocab_size

    def pad_id(self) -> int:
        return self.hf.pad_token_id
