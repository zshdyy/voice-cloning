import argparse
import os
import sys
import traceback

def main():
    parser = argparse.ArgumentParser(description="Run OpenVoice zero-shot voice conversion")
    parser.add_argument("--source", required=True, help="Source audio (wav recommended)")
    parser.add_argument("--target", required=True, help="Target reference audio")
    parser.add_argument("--out", required=True, help="Output wav path")
    parser.add_argument("--ckpt_dir", default="checkpoints_v2", help="OpenVoice checkpoints root")
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    args = parser.parse_args()

    try:
        import torch
        import wavmark
        from openvoice.api import ToneColorConverter
        from openvoice import se_extractor
        from pydub import AudioSegment
    except Exception as e:
        print(f"ERROR: failed to import OpenVoice: {e}", file=sys.stderr)
        return 2

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt_dir = args.ckpt_dir
    converter_dir = os.path.join(ckpt_dir, "converter")
    config_path = os.path.join(converter_dir, "config.json")
    ckpt_path = os.path.join(converter_dir, "checkpoint.pth")

    ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg", "bin", "ffmpeg.exe"))
    if os.path.exists(ffmpeg_path):
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        AudioSegment.converter = ffmpeg_path

    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print(f"ERROR: OpenVoice checkpoints not found in {converter_dir}", file=sys.stderr)
        print("Expected files: config.json and checkpoint.pth", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    try:
        # Avoid HuggingFace download for WavMark; we do not need watermarking.
        class _NoWatermark:
            def to(self, device):
                return self

        wavmark.load_model = lambda *args, **kwargs: _NoWatermark()

        tcc = ToneColorConverter(config_path, device=device)
        tcc.load_ckpt(ckpt_path)
        tcc.watermark_model = None

        # Extract embeddings with VAD enabled for cleaner timbre
        src_se, _ = se_extractor.get_se(args.source, tcc, target_dir=os.path.dirname(args.out), vad=True)
        tgt_se, _ = se_extractor.get_se(args.target, tcc, target_dir=os.path.dirname(args.out), vad=True)

        tcc.convert(
            audio_src_path=args.source,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=args.out,
            tau=0.3,
            message="default",
        )
    except Exception as e:
        print(f"ERROR: OpenVoice conversion failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 4

    print("DONE", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
