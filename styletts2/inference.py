import torch
import yaml
import librosa
from pathlib import Path
from munch import Munch
from nltk.tokenize import word_tokenize
from phonemizer.backend import EspeakBackend

from .models import build_model, load_ASR_models, load_F0_models, load_checkpoint
from .text_utils import TextCleaner
from .utils import recursive_munch, length_to_mask
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


def prepare_dataset(data_dir: str, output_list: str) -> None:
    """Create a data list from an LJSpeech-style dataset."""
    data_dir = Path(data_dir)
    metadata = data_dir / "metadata.csv"
    wav_dir = data_dir / "wavs"

    lines = []
    with open(metadata, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                wav_path = wav_dir / f"{parts[0]}.wav"
                text = parts[1]
                lines.append(f"{wav_path}|{text}|0\n")

    with open(output_list, "w", encoding="utf-8") as out_f:
        out_f.writelines(lines)


class TTSModel:
    def __init__(self, config_path: str, checkpoint_path: str, device: str | None = None, *, language: str = "en-us"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = yaml.safe_load(open(config_path))

        asr_config = self.config.get("ASR_config", False)
        asr_path = self.config.get("ASR_path", False)
        text_aligner = load_ASR_models(asr_path, asr_config)

        f0_path = self.config.get("F0_path", False)
        pitch_extractor = load_F0_models(f0_path)

        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(self.config.get("PLBERT_dir", False))

        self.model = build_model(recursive_munch(self.config["model_params"]),
                                 text_aligner, pitch_extractor, plbert)
        _ = [self.model[k].to(self.device).eval() for k in self.model]

        load_checkpoint(self.model, None, checkpoint_path, load_only_params=True)

        self.sampler = DiffusionSampler(
            self.model["diffusion"].diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        self.phonemizer = EspeakBackend(language=language, preserve_punctuation=True, with_stress=True)
        self.text_cleaner = TextCleaner()

    def synthesize(self, text: str, diffusion_steps: int = 5, embedding_scale: float = 1.0):
        text = text.strip().replace('"', '')
        ps = self.phonemizer.phonemize([text])[0]
        ps = " ".join(word_tokenize(ps))
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model["text_encoder"](tokens, input_lengths, text_mask)
            bert_dur = self.model["bert"](tokens, attention_mask=(~text_mask).int())
            d_en = self.model["bert_encoder"](bert_dur).transpose(-1, -2)

            noise = torch.randn(1, 1, 256).to(self.device)
            s_pred = self.sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model["predictor"].text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model["predictor"].lstm(d)
            duration = self.model["predictor"].duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data)).to(self.device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
            F0_pred, N_pred = self.model["predictor"].F0Ntrain(en, s)
            out = self.model["decoder"](
                (t_en @ pred_aln_trg.unsqueeze(0)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy()
