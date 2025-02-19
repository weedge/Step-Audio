# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import torch
from torch.nn import functional as F

from cosyvoice.utils.common import fade_in_out


class CosyVoiceModel:
    def __init__(
        self,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flow = flow
        self.hift = hift

        # dict used to store session related variable
        self.lock = threading.Lock()  # dict lock
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def token2wav(
        self,
        token,
        prompt_token,
        prompt_feat,
        embedding,
        session_id,
        finalize=False,
        speed=1.0,
        is_flow_cache=False,
        is_hift_cache=False,
    ):
        if is_flow_cache is True and session_id not in self.flow_cache_dict:
            with self.lock:
                self.mel_overlap_dict[session_id] = torch.zeros(1, 80, 0)
                self.flow_cache_dict[session_id] = torch.zeros(1, 80, 0, 2)

        if is_hift_cache is True and session_id not in self.hift_cache_dict:
            with self.lock:
                self.hift_cache_dict[session_id] = None

        tts_mel, flow_cache = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                self.device
            ),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device),
            flow_cache=self.flow_cache_dict[session_id] if is_flow_cache else None,
        )
        self.flow_cache_dict[session_id] = flow_cache if is_flow_cache else None

        # mel overlap fade in out
        if is_flow_cache and self.mel_overlap_dict[session_id].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[session_id], self.mel_window)

        hift_cache_source = None
        if is_hift_cache is True:
            if self.hift_cache_dict[session_id] is not None:
                # append hift cache
                hift_cache_mel, hift_cache_source = (
                    self.hift_cache_dict[session_id]["mel"],
                    self.hift_cache_dict[session_id]["source"],
                )
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                hift_cache_source = torch.zeros(1, 1, 0)

        # keep overlap mel and hift cache
        if finalize is False:
            if is_flow_cache is True:
                self.mel_overlap_dict[session_id] = tts_mel[:, :, -self.mel_overlap_len :]

            tts_mel = tts_mel[:, :, : -self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )

            if is_hift_cache is True:
                if self.hift_cache_dict[session_id] is not None:
                    tts_speech = fade_in_out(
                        tts_speech, self.hift_cache_dict[session_id]["speech"], self.speech_window
                    )
                self.hift_cache_dict[session_id] = {
                    "mel": tts_mel[:, :, -self.mel_cache_len :],
                    "source": tts_source[:, :, -self.source_cache_len :],
                    "speech": tts_speech[:, -self.source_cache_len :],
                }

            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            if speed != 1.0:
                if is_hift_cache is True:
                    assert (
                        self.hift_cache_dict[session_id] is None
                    ), "speed change only support non-stream inference mode"
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")
            tts_speech, tts_source = self.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )
            if is_hift_cache is True and self.hift_cache_dict[session_id] is not None:
                tts_speech = fade_in_out(
                    tts_speech, self.hift_cache_dict[session_id]["speech"], self.speech_window
                )

        return tts_speech
