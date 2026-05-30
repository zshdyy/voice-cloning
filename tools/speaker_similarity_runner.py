import argparse
import json
import math
import os
import sys
import tempfile


MIN_EMBED_SEC = 6.0


def _maybe_extend_audio_for_embedding(audio_path, min_duration_sec=MIN_EMBED_SEC):
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    min_duration_ms = int(float(min_duration_sec) * 1000)

    if duration_ms <= 0:
        raise RuntimeError(f"Audio is empty: {audio_path}")

    if duration_ms >= min_duration_ms:
        return audio_path, None

    repeat_count = max(2, int(math.ceil(min_duration_ms / float(duration_ms))))
    extended = AudioSegment.silent(duration=0)
    for _ in range(repeat_count):
        extended += audio

    fd, temp_path = tempfile.mkstemp(prefix="spk_sim_ext_", suffix=".wav")
    os.close(fd)
    extended.export(temp_path, format="wav")
    return temp_path, temp_path


def _load_openvoice_components(ckpt_dir: str, device_arg: str):
    import torch
    import wavmark
    from openvoice.api import ToneColorConverter
    from pydub import AudioSegment

    if device_arg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    converter_dir = os.path.join(ckpt_dir, "converter")
    config_path = os.path.join(converter_dir, "config.json")
    ckpt_path = os.path.join(converter_dir, "checkpoint.pth")

    ffmpeg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ffmpeg", "bin", "ffmpeg.exe")
    )
    if os.path.exists(ffmpeg_path):
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        AudioSegment.converter = ffmpeg_path

    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"OpenVoice checkpoints not found in {converter_dir}; expected config.json and checkpoint.pth"
        )

    class _NoWatermark:
        def to(self, device):
            return self

    wavmark.load_model = lambda *args, **kwargs: _NoWatermark()

    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(ckpt_path)
    converter.watermark_model = None
    return converter, device


def _embedding_to_vector(embedding):
    import numpy as np
    import torch

    if isinstance(embedding, torch.Tensor):
        arr = embedding.detach().float().cpu().numpy()
    else:
        arr = np.asarray(embedding, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def _cosine_similarity(vec_a, vec_b):
    import numpy as np

    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _normalized_score(cosine_similarity: float):
    score = (cosine_similarity + 1.0) / 2.0 * 100.0
    if score < 0.0:
        return 0.0
    if score > 100.0:
        return 100.0
    return score


def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker similarity between two audio files")
    parser.add_argument("--reference", required=True, help="Target/reference audio")
    parser.add_argument("--candidate", required=True, help="Generated/candidate audio")
    parser.add_argument("--ckpt_dir", default="checkpoints_v2", help="OpenVoice checkpoints root")
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    parser.add_argument("--out_json", default="", help="Optional JSON output path")
    parser.add_argument("--label_reference", default="reference", help="Reference audio label")
    parser.add_argument("--label_candidate", default="candidate", help="Candidate audio label")
    args = parser.parse_args()

    try:
        from openvoice import se_extractor
    except Exception as e:
        print(f"ERROR: failed to import OpenVoice speaker extractor: {e}", file=sys.stderr)
        return 2

    try:
        converter, device = _load_openvoice_components(args.ckpt_dir, args.device)
    except Exception as e:
        print(f"ERROR: failed to load OpenVoice converter: {e}", file=sys.stderr)
        return 3

    work_dir = os.path.dirname(os.path.abspath(args.out_json)) if args.out_json else os.path.dirname(os.path.abspath(args.candidate))
    os.makedirs(work_dir, exist_ok=True)

    ref_temp = None
    cand_temp = None
    try:
        reference_for_se, ref_temp = _maybe_extend_audio_for_embedding(args.reference)
        candidate_for_se, cand_temp = _maybe_extend_audio_for_embedding(args.candidate)

        ref_embedding, _ = se_extractor.get_se(reference_for_se, converter, target_dir=work_dir, vad=True)
        cand_embedding, _ = se_extractor.get_se(candidate_for_se, converter, target_dir=work_dir, vad=True)

        ref_vector = _embedding_to_vector(ref_embedding)
        cand_vector = _embedding_to_vector(cand_embedding)
        cosine_similarity = _cosine_similarity(ref_vector, cand_vector)
        similarity_score = _normalized_score(cosine_similarity)

        result = {
            "reference_audio": os.path.abspath(args.reference),
            "candidate_audio": os.path.abspath(args.candidate),
            "reference_label": args.label_reference,
            "candidate_label": args.label_candidate,
            "embedding_backend": "OpenVoice se_extractor",
            "device": device,
            "cosine_similarity": round(cosine_similarity, 6),
            "similarity_score": round(similarity_score, 2),
        }

        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"ERROR: speaker similarity evaluation failed: {e}", file=sys.stderr)
        return 4
    finally:
        for temp_path in [ref_temp, cand_temp]:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
