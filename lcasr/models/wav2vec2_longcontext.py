import torch, torch.nn as nn
from typing import Optional

from transformers import Wav2Vec2Model
from transformers.masking_utils import create_bidirectional_mask

from lcasr.components.decoder import ASRLinearSCDecoder
from lcasr.models.base import BaseModel


class Wav2Vec2CTCLongContext(BaseModel):
    '''
    Wraps a pretrained HF Wav2Vec2Model's feature_projection + Transformer encoder for
    CTC training under the same chunked sequence-length-warmup curriculum used by
    SCConformerXL (exp/train.py + lcasr.utils.scheduling.SequenceWarmupManager).

    The CNN feature_extractor is NOT included here: it is frozen and run once offline
    per collection by scripts/prepare_ainu_wav2vec2_longcontext.py, so `audio_signal`
    is already the precomputed (batch, conv_dim, time) CNN output. This avoids both the
    CNN's receptive-field boundary effects at chunk edges and re-running a frozen module
    on every step.

    Unlike SCConformerXL, `cached_kvs`/`cached_kv_lengths` are accepted (to keep the
    exp/train.py call signature identical) but never used: SCConformerXL does not
    actually chain KV-caches across chunks either (see sconformer_xl.py's own
    "remove caching stuff as it is not used anymore" TODO), so there is nothing to port.
    '''

    def __init__(
        self,
        vocab_size: int,
        base_model: str = 'karolnowakowski/wav2vec2-large-xlsr-53-pretrain-ain',
        decoder_norm: bool = True,
        self_conditioning: bool = True,
        attn_implementation: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()

        w2v2 = Wav2Vec2Model.from_pretrained(base_model, attn_implementation=attn_implementation)

        self.config = w2v2.config
        self.d_model = w2v2.config.hidden_size
        self.self_conditioning = self_conditioning
        self.stable_layer_norm = bool(w2v2.config.do_stable_layer_norm)
        # audio_signal is already the precomputed CNN output at its native frame
        # rate; there is no further subsampling inside this model (see
        # lcasr/eval/utils.py:fetch_logits, which reads this for eval chunking).
        self.subsampling_factor = 1

        # only feature_projection + the transformer encoder are kept; feature_extractor
        # (the CNN) is precomputed offline and is not part of this module.
        self.feature_projection = w2v2.feature_projection
        self.pos_conv_embed = w2v2.encoder.pos_conv_embed
        self.encoder_dropout = w2v2.encoder.dropout
        self.encoder_layer_norm = w2v2.encoder.layer_norm
        self.layers = w2v2.encoder.layers

        self.decoder = ASRLinearSCDecoder(
            d_model=self.d_model,
            vocab_size=vocab_size,
            norm=decoder_norm,
            norm_fn=nn.LayerNorm,
        )

        # follows SCConformerXL's convention: whitelist -> weight decay applied,
        # blacklist -> no weight decay (see BaseModel.get_param_groups)
        self.whitelist_weight_decay_modules = (nn.LayerNorm, nn.GroupNorm)
        self.blacklist_weight_decay_modules = (nn.Linear, nn.Conv1d)

    def forward(
        self,
        audio_signal: torch.Tensor,  # (batch, conv_dim, time) precomputed wav2vec2 CNN output
        length: Optional[torch.Tensor] = None,
        cached_kvs=None,  # unused, see class docstring
        cached_kv_lengths=None,  # unused, see class docstring
        return_logits: bool = False,
        skip_vocab_projection: bool = False,
    ):
        decoder = self.decoder
        audio_signal = torch.transpose(audio_signal, 1, 2)  # -> (batch, time, conv_dim)
        max_len = audio_signal.size(1)

        if length is None:
            length = torch.tensor([max_len] * audio_signal.size(0), device=audio_signal.device)

        hidden_states, _ = self.feature_projection(audio_signal)

        valid_mask = None
        if length.max() != length.min():
            valid_mask = torch.arange(max_len, device=hidden_states.device).expand(hidden_states.size(0), max_len) < length.unsqueeze(1)
            hidden_states = hidden_states * valid_mask.unsqueeze(-1)  # zero out padded positions, matches HF's native encoder

        # builds whatever mask representation the active attention backend
        # (sdpa/eager/flash_attention_2) actually expects; a hand-built mask is not
        # a stable contract across transformers versions (this changed between the
        # 4.x and 5.x lines we tested against).
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=valid_mask,
        )

        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        if not self.stable_layer_norm:  # base wav2vec2: norm before layers
            hidden_states = self.encoder_layer_norm(hidden_states)
        hidden_states = self.encoder_dropout(hidden_states)

        interim_posteriors = []
        for lth, layer in enumerate(self.layers):
            layer_out = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if lth != len(self.layers) - 1 and self.self_conditioning:
                interim_post = torch.nn.functional.softmax(decoder(x=hidden_states, logits=True), dim=-1)
                interim_posteriors.append(interim_post)
                hidden_states = decoder.integrate_projections(hidden_states, decoder.project_back(interim_post))

        if self.stable_layer_norm:  # xlsr/large wav2vec2: norm after all layers
            hidden_states = self.encoder_layer_norm(hidden_states)

        if skip_vocab_projection:
            return {'hidden_states': hidden_states, 'length': length}

        final_posts = decoder(x=hidden_states, logits=return_logits)
        return {'final_posteriors': final_posts, 'length': length, 'interim_posteriors': interim_posteriors}
