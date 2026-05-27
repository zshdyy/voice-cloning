# Project Handoff

This project uses two Python environments:

1) Single-audio WORLD mode (local venv in this repo)
2) Voice clone mode (external Python environment with Coqui TTS)

## 1) Single-audio WORLD environment

From the project root:

```cmd
python -m venv voice_env_clean
voice_env_clean\Scripts\activate
pip install -r requirements.txt
```

Start GUI:

```cmd
run_gui.bat
```

## 2) Voice clone environment (external)

Install Coqui TTS in a separate Python 3.10 environment (example: conda env `tts_py310`).

Then update env_map.json to point to your Python:

```json
{
  "声线克隆（双音频）": "C:\\REPLACE_WITH_YOUR_TTS_ENV\\python.exe"
}
```

After that, the GUI will run clone mode via tools/clone_runner.py in the external env.

## Notes

- Do NOT package or copy any virtual environment folders (voice_env_clean, voice_env, .venv*).
- This repo already contains ffmpeg/bin for format conversion.
- If you move the project, relative paths still work. Only env_map.json must be updated.
