import sys
import os
import time
import tempfile
import subprocess
from collections import deque
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pyworld as pw

try:
    import noisereduce as nr
    _HAVE_NOIREDUCE = True
except Exception:
    nr = None
    _HAVE_NOIREDUCE = False

try:
    import PyQt5
    _qt_plugins_dir = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_qt_plugins_dir):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _qt_plugins_dir)
        os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins_dir)
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QSlider,
    QProgressBar,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath

try:
    import simpleaudio as sa
    _HAVE_SIMPLEAUDIO = True
except Exception:
    sa = None
    _HAVE_SIMPLEAUDIO = False

try:
    import winsound
    _HAVE_WINSOUND = True
except Exception:
    winsound = None
    _HAVE_WINSOUND = False

try:
    from config import MIN_AUDIO_LENGTH, MAX_AUDIO_LENGTH, SUPPORTED_FORMATS
except Exception:
    MIN_AUDIO_LENGTH = 3.0
    MAX_AUDIO_LENGTH = 300.0
    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


def _supported_audio_text():
    return ", ".join(SUPPORTED_FORMATS)


def _is_supported_audio(path):
    return any(path.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


def _audio_duration_seconds(path):
    """返回音频时长（秒）。失败时抛出异常。"""
    # 先尝试 soundfile / librosa 的快速路径；失败后用解码后的帧长兜底
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        try:
            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            # 最后尝试使用本地 ffprobe/ffmpeg 解析 duration（优先项目内 ffmpeg）
            try:
                import subprocess, re
                # 先查找项目内 ffmpeg/ffprobe
                local_ffprobe = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffprobe.exe")
                local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
                cmd = None
                if os.path.exists(local_ffprobe):
                    cmd = [local_ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
                elif os.path.exists(local_ffmpeg):
                    # parse ffmpeg -i stderr
                    cmd = [local_ffmpeg, "-i", path]
                else:
                    # try system ffprobe/ffmpeg in PATH
                    from shutil import which
                    ffp = which("ffprobe") or which("ffmpeg")
                    if ffp:
                        if ffp.endswith("ffprobe") or ffp.lower().endswith("ffprobe.exe"):
                            cmd = [ffp, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
                        else:
                            cmd = [ffp, "-i", path]

                if cmd is None:
                    raise RuntimeError("未检测到可用的 ffprobe/ffmpeg，用于解析容器时长。请安装 ffmpeg 或把项目目录下的 ffmpeg 放在 ffmpeg/bin。")

                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                # first try to find a plain number (ffprobe output)
                m = re.search(r"^(\d+\.\d+)$", out.strip(), re.M)
                if m:
                    return float(m.group(1))
                # look for Duration: HH:MM:SS.ms
                m2 = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
                if m2:
                    hh = int(m2.group(1)); mm = int(m2.group(2)); ss = float(m2.group(3))
                    return float(hh * 3600 + mm * 60 + ss)
                raise RuntimeError(f"无法从 ffmpeg 输出解析时长。输出: {out[:300]}")
            except Exception as e:
                # 最后尝试完整解码（最慢）
                try:
                    y, sr = librosa.load(path, sr=None, mono=False)
                    if y is None:
                        raise RuntimeError("无法读取音频数据。")
                    return float(len(y)) / float(sr)
                except Exception:
                    raise RuntimeError(f"无法读取音频时长: {e}")


def _duration_status(path, role):
    duration = _audio_duration_seconds(path)
    warnings = []
    if duration < MIN_AUDIO_LENGTH:
        warnings.append(f"{role} 音频太短（{duration:.2f}s），建议至少 {MIN_AUDIO_LENGTH:.0f}s。")
    if duration > MAX_AUDIO_LENGTH:
        warnings.append(f"{role} 音频过长（{duration:.2f}s），建议不超过 {MAX_AUDIO_LENGTH:.0f}s。")
    return duration, warnings


def _convert_to_temp_wav(path, prefix):
    """将任意输入音频转成 16k 单声道临时 WAV，供 TTS 声线转换使用。"""
    # For non-WAV inputs, prefer ffmpeg to avoid audioread/mpg123 errors.
    ext = os.path.splitext(path)[1].lower()
    prefer_ffmpeg = ext != ".wav"

    def _ffmpeg_convert():
        # 优先使用项目内的 ffmpeg/bin/ffmpeg.exe，其次使用 PATH 中的 ffmpeg
        local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
        ffmpeg_cmd = None
        if os.path.exists(local_ffmpeg):
            ffmpeg_cmd = local_ffmpeg
        else:
            from shutil import which
            ffp = which("ffmpeg")
            if ffp:
                ffmpeg_cmd = ffp

        if ffmpeg_cmd is None:
            raise RuntimeError("无法读取音频且未检测到 ffmpeg，无法生成临时 WAV。请安装 ffmpeg 或在项目中放置 ffmpeg/bin/ffmpeg.exe。")

        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
        os.close(fd)
        # On Windows, some ffmpeg builds have issues with non-ANSI paths; try to use short (8.3) path
        try:
            if os.name == 'nt':
                import ctypes
                buf = ctypes.create_unicode_buffer(260)
                r = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
                if r:
                    ff_input = buf.value
                else:
                    ff_input = path
            else:
                ff_input = path
        except Exception:
            ff_input = path

        cmd = [ffmpeg_cmd, "-y", "-i", ff_input, "-ar", "16000", "-ac", "1", "-vn", temp_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(temp_path):
            # 清理并抛出
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg 转换失败 (rc={proc.returncode})，请检查 ffmpeg 可用性。stderr: {proc.stderr.decode(errors='ignore')}")
        return temp_path

    if prefer_ffmpeg:
        return _ffmpeg_convert()

    try:
        y, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
        os.close(fd)
        sf.write(temp_path, y, 16000)
        return temp_path
    except Exception:
        return _ffmpeg_convert()


def apply_energy_vad(y, fs, frame_ms=20, threshold_ratio=0.05):
    """
    [纯手写] 基于短时能量的语音活动检测 (VAD) 与软降噪
    :param y: 输入音频信号 (1D numpy array, float64)
    :param fs: 采样率
    :param frame_ms: 帧长 (毫秒)
    :param threshold_ratio: 能量阈值比例 (越小越激进，越大越温和)
    :return: (y_clean, mask_expanded) - 处理后的音频和掩码
    """
    frame_length = int(fs * (frame_ms / 1000.0))
    hop_length = frame_length // 2
    
    # 1. 计算每一帧的短时能量 (STE)
    energies = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i : i + frame_length]
        energies.append(np.sum(frame ** 2))
    
    if len(energies) == 0:
        return y, np.zeros_like(y)
    
    energies = np.array(energies)
    
    # 2. 设定自适应阈值 (最大能量的 threshold_ratio 倍)
    threshold = np.max(energies) * threshold_ratio
    
    # 3. 生成二值掩码: 大于阈值为 1(人声), 小于为 0(噪音)
    mask = (energies > threshold).astype(float)
    
    # 4. 平滑掩码，防止声音断断续续 (Hangover 机制)
    smoothed_mask = np.copy(mask)
    for i in range(1, len(mask) - 1):
        if mask[i-1] == 1 and mask[i+1] == 1:
            smoothed_mask[i] = 1.0
    
    # 5. 将掩码还原回原始信号长度
    mask_expanded = np.repeat(smoothed_mask, hop_length)
    pad_length = len(y) - len(mask_expanded)
    if pad_length > 0:
        mask_expanded = np.pad(mask_expanded, (0, pad_length), 'edge')
    else:
        mask_expanded = mask_expanded[:len(y)]
    
    # 6. 施加软门限 (Soft Gate)
    # 人声部分保留 100%, 噪音部分保留 10% (不直接归零，听感更自然)
    final_gain = mask_expanded * 0.9 + 0.1
    y_clean = y * final_gain
    
    return y_clean, mask_expanded


def apply_phase2_noise_reduction(y, fs, mode="标准"):
    """第二阶段：基于 noisereduce 的谱减噪，提供三档力度。"""
    if not _HAVE_NOIREDUCE:
        return y

    mode = (mode or "标准").strip()
    settings = {
        "温和": {
            "stationary": False,
            "prop_decrease": 0.55,
            "time_constant_s": 1.5,
            "freq_mask_smooth_hz": 300,
            "time_mask_smooth_ms": 35,
            "thresh_n_mult_nonstationary": 2.2,
        },
        "标准": {
            "stationary": False,
            "prop_decrease": 0.75,
            "time_constant_s": 1.0,
            "freq_mask_smooth_hz": 500,
            "time_mask_smooth_ms": 50,
            "thresh_n_mult_nonstationary": 2.0,
        },
        "强力": {
            "stationary": False,
            "prop_decrease": 1.0,
            "time_constant_s": 0.8,
            "freq_mask_smooth_hz": 700,
            "time_mask_smooth_ms": 70,
            "thresh_n_mult_nonstationary": 1.7,
        },
    }.get(mode, {
        "stationary": False,
        "prop_decrease": 0.75,
        "time_constant_s": 1.0,
        "freq_mask_smooth_hz": 500,
        "time_mask_smooth_ms": 50,
        "thresh_n_mult_nonstationary": 2.0,
    })

    y_in = np.asarray(y, dtype=np.float64)
    if y_in.ndim > 1:
        y_in = np.mean(y_in, axis=1)

    with np.errstate(all="ignore"):
        y_out = nr.reduce_noise(y=y_in, sr=int(fs), **settings)

    if not np.isfinite(y_out).all():
        return y_in

    return np.asarray(y_out, dtype=np.float64)


def _analyze_and_convert(x, fs, strength=1.0):
    def _inner(x_in, fs_in, strength=1.0):
        _f0, t = pw.dio(x_in, fs_in)
        f0 = pw.stonemask(x_in, _f0, t, fs_in)
        sp = pw.cheaptrick(x_in, f0, t, fs_in)
        ap = pw.d4c(x_in, f0, t, fs_in)

        valid_f0 = f0[f0 > 0]
        if len(valid_f0) == 0:
            raise RuntimeError("未检测到有效的人声基频。")

        median_f0 = np.median(valid_f0)
        threshold = 165.0
        if median_f0 < threshold:
            detected = "男声"
            pitch_ratio = 220.0 / median_f0
            formant_ratio = 1.18
        else:
            detected = "女声"
            pitch_ratio = 120.0 / median_f0
            formant_ratio = 0.85

        # 生成修改后的 F0 与频谱
        modified_f0 = f0 * pitch_ratio
        modified_sp = np.zeros_like(sp)
        num_bins = sp.shape[1]
        old_freq_axis = np.arange(num_bins)
        for i in range(sp.shape[0]):
            new_freq_axis = old_freq_axis / formant_ratio
            modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])

        # 合成原始音与修改后音，按 strength 做线性混合，strength=1.0 完全修改
        y_orig = pw.synthesize(f0, sp, ap, fs_in)
        y_mod = pw.synthesize(modified_f0, modified_sp, ap, fs_in)

        strength = float(strength)
        strength = max(0.0, min(1.0, strength))
        y = y_mod * strength + y_orig * (1.0 - strength)
        return y, f0, modified_f0, median_f0, detected

    return _inner(x, fs, strength=float(strength))


def plot_f0_histogram(input_path, save_path, bins=50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = librosa.load(input_path, sr=None, dtype=np.float64)
    _, f0_before, f0_after, _, _ = _analyze_and_convert(x, fs)

    valid_f0_before = f0_before[f0_before > 0]
    valid_f0_after = f0_after[f0_after > 0]
    if len(valid_f0_before) == 0 or len(valid_f0_after) == 0:
        raise RuntimeError("未检测到有效基频，无法绘制直方图。")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(valid_f0_before, bins=bins, color="#FF9800", edgecolor="#222", alpha=0.7)
    ax1.set_xlabel("F0 (Hz)")
    ax1.set_ylabel("Count")
    ax1.set_title("Before Conversion", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(valid_f0_after, bins=bins, color="#4CAF50", edgecolor="#222", alpha=0.7)
    ax2.set_xlabel("F0 (Hz)")
    ax2.set_ylabel("Count")
    ax2.set_title("After Conversion", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"F0 Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_mel_spectrogram_comparison(input_path, converted_path, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = librosa.load(input_path, sr=None, dtype=np.float32)
    y, fs2 = librosa.load(converted_path, sr=fs, dtype=np.float32)
    if fs2 != fs:
        raise RuntimeError("采样率不一致，无法绘制 Mel 对比图。")

    n_fft = 1024
    hop_length = 256
    n_mels = 128

    mel_before = librosa.feature.melspectrogram(
        y=x, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    mel_after = librosa.feature.melspectrogram(
        y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )

    mel_before_db = librosa.power_to_db(mel_before, ref=np.max)
    mel_after_db = librosa.power_to_db(mel_after, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    img1 = librosa.display.specshow(
        mel_before_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[0],
    )
    axes[0].set_title("Mel Spectrogram - Before", fontweight="bold")

    librosa.display.specshow(
        mel_after_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[1],
    )
    axes[1].set_title("Mel Spectrogram - After", fontweight="bold")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), format="%+2.0f dB", shrink=0.85)
    cbar.set_label("dB")
    fig.suptitle(f"Mel Spectrogram Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


class VoiceConverterThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, enable_vad=True, threshold_ratio=0.05, frame_ms=20, enable_phase2=False, phase2_mode="标准", strength=1.0):
        super().__init__()
        self.input_path = input_path
        self.enable_vad = enable_vad
        self.threshold_ratio = threshold_ratio
        self.frame_ms = frame_ms
        self.enable_phase2 = enable_phase2
        self.phase2_mode = phase2_mode
        self.strength = float(strength)

    def run(self):
        try:
            self.log_signal.emit(f"正在加载音频: {os.path.basename(self.input_path)}")
            temp_wav = None
            try:
                x, fs = librosa.load(self.input_path, sr=None, dtype=np.float64)
            except Exception:
                self.log_signal.emit("⚠️ 直接解码失败，尝试转为临时 WAV...")
                temp_wav = _convert_to_temp_wav(self.input_path, "vc_temp_")
                x, fs = librosa.load(temp_wav, sr=None, dtype=np.float64)
            
            # ====== 核心拦截器：VAD 条件分支 ======
            mask = None
            if self.enable_vad:
                self.log_signal.emit(f"🔕 [VAD 启动] 正在进行动态能量分析与降噪 (阈值: {self.threshold_ratio*100:.0f}%, 帧长: {self.frame_ms}ms)...")
                x, mask = apply_energy_vad(x, fs, frame_ms=self.frame_ms, threshold_ratio=self.threshold_ratio)
                self.log_signal.emit("✅ 噪音与静音段抑制完成！")
            else:
                self.log_signal.emit("⚠️ [VAD 关闭] 原始音频直通 WORLD 声码器。")

            if self.enable_phase2:
                if _HAVE_NOIREDUCE:
                    self.log_signal.emit(f"🧼 [第二阶段] 启动谱减噪，模式: {self.phase2_mode}...")
                    x = apply_phase2_noise_reduction(x, fs, mode=self.phase2_mode)
                    self.log_signal.emit("✅ 第二阶段谱减噪完成！")
                else:
                    self.log_signal.emit("⚠️ 第二阶段降噪未启用：当前环境缺少 noisereduce。")
            # ===================================
            
            self.log_signal.emit("正在使用 WORLD 提取特征...")

            y, _, _, median_f0, detected = _analyze_and_convert(x, fs, strength=self.strength)
            self.log_signal.emit(f"基频中位数: {median_f0:.1f} Hz")
            self.log_signal.emit(f"识别结果: {detected}")
            
            # ====== 防爆音规范化：防止波形幅度溢出 ======
            peak = np.max(np.abs(y))
            if peak > 0.95:
                self.log_signal.emit("🔊 正在进行音频幅度规范化 (Normalization)...")
                y = y * (0.95 / peak)
            # =======================================

            output_dir = os.path.dirname(self.input_path)
            base_name = os.path.basename(self.input_path)
            stem, ext = os.path.splitext(base_name)
            if ext.lower() != ".wav":
                output_name = f"converted_{stem}.wav"
            else:
                output_name = f"converted_{base_name}"
            output_path = os.path.join(output_dir, output_name)
            sf.write(output_path, y, fs)

            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

            # 如果存在 VAD 掩码，保存到 sidecar 文件，便于播放时混音
            try:
                if mask is not None:
                    np.save(output_path + ".mask.npy", mask)
            except Exception:
                pass

            self.log_signal.emit("✅ 转换成功")
            self.finished_signal.emit(output_path)
        except Exception as e:
            self.log_signal.emit(f"❌ 发生异常: {e}")
            self.finished_signal.emit("")


class VoiceCloneThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, source_path, target_paths, engine="FreeVC"):
        super().__init__()
        self.source_path = source_path
        self.engine = engine
        # target_paths 可为列表（多参考样本）或单个路径
        if isinstance(target_paths, (list, tuple)):
            self.target_paths = list(target_paths)
        else:
            self.target_paths = [target_paths]
        self._temp_files = []

    def _prepare_temp_wav(self, path, prefix):
        # path 可能是单路径，也可能是 list（多个目标）
        if isinstance(path, (list, tuple)):
            # 合并多个参考文件为单个临时 wav（顺序拼接）
            combined = []
            sr_target = 16000
            for p in path:
                try:
                    seg_temp = _convert_to_temp_wav(p, f"{prefix}seg_")
                    self._temp_files.append(seg_temp)
                    y, sr = sf.read(seg_temp, dtype="float32")
                    if y is None:
                        continue
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != sr_target:
                        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                    combined.append(y)
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 参考音频处理失败: {os.path.basename(p)} ({e})")
                    # 跳过有问题的文件
                    pass
            if len(combined) == 0:
                raise RuntimeError("无法从参考音频生成临时文件：无有效文件。")
            y_all = np.concatenate(combined, axis=0)
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix='.wav')
            os.close(fd)
            sf.write(temp_path, y_all, sr_target)
            self._temp_files.append(temp_path)
            return temp_path
        else:
            temp_path = _convert_to_temp_wav(path, prefix)
            self._temp_files.append(temp_path)
            return temp_path

    def run(self):
        source_temp = None
        target_temp = None
        try:
            self.log_signal.emit(f"正在加载源音频: {os.path.basename(self.source_path)}")
            # 如果有多个参考，显示第一个作为代表
            rep_target = self.target_paths[0] if len(self.target_paths) > 0 else ""
            self.log_signal.emit(f"正在加载目标音频: {os.path.basename(rep_target)}")

            source_temp = self._prepare_temp_wav(self.source_path, "voice_source_")
            target_temp = self._prepare_temp_wav(self.target_paths, "voice_target_")

            # 首先尝试使用外部映射到的 Python 环境来运行声线克隆（避免在 GUI env 安装 heavy 依赖）
            try:
                map_path = os.path.join(os.path.dirname(__file__), 'env_map.json')
                python_exec = None
                openvoice_ckpt = None
                if os.path.exists(map_path):
                    import json
                    with open(map_path, 'r', encoding='utf-8') as f:
                        emap = json.load(f)
                        if self.engine == "OpenVoice":
                            python_exec = emap.get('声线克隆（OpenVoice）')
                            openvoice_ckpt = emap.get('OpenVoice_Checkpoints')
                        else:
                            python_exec = emap.get('声线克隆（双音频）')

                if python_exec and os.path.exists(python_exec):
                    self.log_signal.emit(f"📡 使用外部 Python: {python_exec} 执行声线克隆")
                    if self.engine == "OpenVoice":
                        runner = os.path.join(os.path.dirname(__file__), 'tools', 'openvoice_runner.py')
                    else:
                        runner = os.path.join(os.path.dirname(__file__), 'tools', 'clone_runner.py')
                    # 组织输出路径
                    output_dir = os.path.dirname(self.source_path)
                    source_base = os.path.splitext(os.path.basename(self.source_path))[0]
                    rep_target = self.target_paths[0] if len(self.target_paths) > 0 else "target"
                    target_base = os.path.splitext(os.path.basename(rep_target))[0]
                    output_path = os.path.join(output_dir, f"vc_{source_base}_to_{target_base}.wav")

                    cmd = [python_exec, runner, '--source', source_temp, '--target', target_temp, '--out', output_path]
                    if self.engine == "OpenVoice" and openvoice_ckpt:
                        cmd.extend(["--ckpt_dir", openvoice_ckpt])
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    # stream stdout/stderr back to GUI
                    for line in proc.stdout:
                        self.log_signal.emit(line.rstrip())
                    err = proc.stderr.read()
                    if err:
                        for l in err.splitlines():
                            self.log_signal.emit(l)

                    rc = proc.wait()
                    if rc == 0 and os.path.exists(output_path):
                        self.log_signal.emit("✅ 声线转换成功 (外部运行)")
                        self.finished_signal.emit(output_path)
                        return
                    else:
                        self.log_signal.emit(f"⚠️ 外部克隆进程失败，rc={rc}")

            except Exception as e:
                self.log_signal.emit(f"⚠️ 外部克隆尝试出错: {e}")

            # 外部运行不可用或失败，尝试在当前环境内直接使用 TTS（如果可用）
            try:
                from TTS.api import TTS
                import torch
                model_name = "voice_conversion_models/multilingual/vctk/freevc24"
                self.log_signal.emit(f"在当前环境加载声线转换模型: {model_name}")
                t0 = time.time()
                tts = TTS(model_name, gpu=torch.cuda.is_available())
                t1 = time.time()
                self.log_signal.emit("正在执行声线转换...")

                output_dir = os.path.dirname(self.source_path)
                source_base = os.path.splitext(os.path.basename(self.source_path))[0]
                rep_target = self.target_paths[0] if len(self.target_paths) > 0 else "target"
                target_base = os.path.splitext(os.path.basename(rep_target))[0]
                output_path = os.path.join(output_dir, f"vc_{source_base}_to_{target_base}.wav")

                tts.voice_conversion_to_file(
                    source_wav=source_temp,
                    target_wav=target_temp,
                    file_path=output_path,
                )
                t2 = time.time()

                self.log_signal.emit(f"模型加载耗时: {t1 - t0:.2f}s")
                self.log_signal.emit(f"声线转换耗时: {t2 - t1:.2f}s")
                self.log_signal.emit(f"总耗时: {t2 - t0:.2f}s")
                self.log_signal.emit("✅ 声线转换成功 (当前环境)")
                self.finished_signal.emit(output_path)
                return
            except Exception as e:
                self.log_signal.emit(f"❌ 声线转换失败: {e}")
                self.finished_signal.emit("")
        finally:
            for temp_path in self._temp_files:
                try:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass


class PlotF0Thread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, save_path):
        super().__init__()
        self.input_path = input_path
        self.save_path = save_path

    def run(self):
        try:
            self.log_signal.emit(f"绘制 F0 对比图: {os.path.basename(self.input_path)}")
            plot_f0_histogram(self.input_path, self.save_path)
            self.log_signal.emit("F0 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            self.log_signal.emit(f"F0 绘图失败: {e}")
            self.finished_signal.emit("")


class PlotMelThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, converted_path, save_path):
        super().__init__()
        self.input_path = input_path
        self.converted_path = converted_path
        self.save_path = save_path

    def run(self):
        try:
            self.log_signal.emit("绘制 Mel 语谱图对比...")
            plot_mel_spectrogram_comparison(self.input_path, self.converted_path, self.save_path)
            self.log_signal.emit("Mel 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            self.log_signal.emit(f"Mel 绘图失败: {e}")
            self.finished_signal.emit("")


class DragDropLabel(QLabel):
    file_dropped = pyqtSignal(str)
    file_list_dropped = pyqtSignal(list)

    def __init__(self, prompt_text=None):
        super().__init__()
        hint = f"\n\n拖动音频文件到这里\n支持格式：{_supported_audio_text()}\n（会自动转换为 WAV）\n\n"
        self.setText(prompt_text or hint)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #555;
            }
            """
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            paths = [u.toLocalFile() for u in urls]
            valid_paths = [p for p in paths if _is_supported_audio(p)]
            if not valid_paths:
                event.ignore()
                self.setStyleSheet(
                    "border: 2px dashed #aaa; background-color: #fff5f5; font-size: 16px;"
                )
                return
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #4CAF50; background-color: #e8f5e9; font-size: 16px;"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")

    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")
        urls = event.mimeData().urls()
        if urls:
            # 支持多文件拖放
            paths = [u.toLocalFile() for u in urls]
            valid_paths = [p for p in paths if _is_supported_audio(p)]
            if len(valid_paths) == 0:
                self.setText(f"\n\n请拖入支持的音频格式\n支持格式：{_supported_audio_text()}\n\n")
                QMessageBox.warning(
                    self.window(),
                    "格式不支持",
                    f"请拖入以下格式的音频文件：\n{_supported_audio_text()}\n\n当前拖入的文件没有被识别为支持的音频格式。",
                )
                return
            if len(valid_paths) == 1:
                self.file_dropped.emit(valid_paths[0])
            else:
                # 多文件回调
                self.file_list_dropped.emit(valid_paths)


class WaveformStrip(QWidget):
    def __init__(self, parent=None, history_size=240):
        super().__init__(parent)
        self.history = deque([0.0] * history_size, maxlen=history_size)
        self.setMinimumHeight(56)
        self.setMaximumHeight(72)

    def push_level(self, level):
        level = max(0.0, min(1.0, float(level)))
        self.history.append(level)
        self.update()

    def reset(self):
        self.history = deque([0.0] * self.history.maxlen, maxlen=self.history.maxlen)
        self.update()

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        mid = h / 2.0
        pad = 4.0

        painter.fillRect(self.rect(), QColor("#101418"))

        grid_pen = QPen(QColor("#25303a"), 1)
        painter.setPen(grid_pen)
        painter.drawLine(0, int(mid), w, int(mid))

        line_pen = QPen(QColor("#00e676"), 1.8)
        painter.setPen(line_pen)

        values = list(self.history)
        if len(values) < 2:
            return

        step_x = w / float(len(values) - 1)
        amp = max(mid - pad, 1.0)

        upper = QPainterPath()
        lower = QPainterPath()
        for i, v in enumerate(values):
            x = i * step_x
            y_up = mid - v * amp
            y_dn = mid + v * amp
            if i == 0:
                upper.moveTo(x, y_up)
                lower.moveTo(x, y_dn)
            else:
                upper.lineTo(x, y_up)
                lower.lineTo(x, y_dn)

        painter.drawPath(upper)
        painter.drawPath(lower)


class VoiceChangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_filepath = None
        self.target_filepaths = []
        self.converted_filepath = None
        self._temp_play_path = None

        self.play_obj = None
        self.play_proc = None
        self.play_data = None
        self.play_rate = 0
        self.play_channels = 0
        self.play_total_ms = 0
        self.play_start_ms = 0
        self.play_start_epoch = 0.0
        self.current_play_path = None

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(80)
        self.play_timer.timeout.connect(self._on_playback_timer)

        self.mode_names = ["WORLD 单音频变声", "声线克隆（双音频）"]

        self.initUI()

    def initUI(self):
        self.setWindowTitle("智能性别语音转换系统 - 深度增强版")
        self.resize(760, 560)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        mode_row = QHBoxLayout()
        mode_label = QLabel("工作模式:")
        mode_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 80px;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(self.mode_names)
        self.combo_mode.setCurrentIndex(0)
        self.combo_mode.setStyleSheet("QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 220px; }")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.combo_mode)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.drop_label = DragDropLabel("\n\n拖动源音频文件到这里\n\n")
        self.drop_label.file_dropped.connect(self.load_file)
        layout.addWidget(self.drop_label)

        # 源文件选择按钮行
        src_btn_row = QHBoxLayout()
        self.btn_choose_source = QPushButton("选择源文件")
        self.btn_choose_source.setStyleSheet("font-size:13px; padding:6px;")
        self.btn_choose_source.clicked.connect(self.choose_source_file)
        src_btn_row.addWidget(self.btn_choose_source)
        src_btn_row.addStretch(1)
        layout.addLayout(src_btn_row)

        self.source_info_label = QLabel("源音频: 未选择")
        self.source_info_label.setStyleSheet("font-size: 12px; color: #666; padding: 4px 0px;")
        layout.addWidget(self.source_info_label)

        self.target_title_label = QLabel("目标音频（声线克隆模式）")
        self.target_title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #333; padding: 8px 0px 4px 0px;")
        layout.addWidget(self.target_title_label)

        self.target_drop_label = DragDropLabel("\n\n拖动目标音频文件到这里\n\n")
        self.target_drop_label.file_dropped.connect(self.load_target_file)
        self.target_drop_label.file_list_dropped.connect(self.load_target_files)
        layout.addWidget(self.target_drop_label)

        # 目标文件选择按钮行
        tgt_btn_row = QHBoxLayout()
        self.btn_choose_target = QPushButton("选择目标文件")
        self.btn_choose_target.setStyleSheet("font-size:13px; padding:6px;")
        self.btn_choose_target.clicked.connect(self.choose_target_file)
        tgt_btn_row.addWidget(self.btn_choose_target)
        tgt_btn_row.addStretch(1)
        layout.addLayout(tgt_btn_row)

        self.target_info_label = QLabel("目标音频: 未选择")
        self.target_info_label.setStyleSheet("font-size: 12px; color: #666; padding: 4px 0px;")
        layout.addWidget(self.target_info_label)
        
        clone_engine_row = QHBoxLayout()
        self.clone_engine_label = QLabel("克隆引擎:")
        self.clone_engine_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        self.combo_clone_engine = QComboBox()
        self.combo_clone_engine.addItems(["FreeVC", "OpenVoice"])
        self.combo_clone_engine.setCurrentText("FreeVC")
        self.combo_clone_engine.setStyleSheet("QComboBox { font-size: 13px; padding: 4px; min-width: 160px; }")
        clone_engine_row.addWidget(self.clone_engine_label)
        clone_engine_row.addWidget(self.combo_clone_engine)
        clone_engine_row.addStretch(1)
        layout.addLayout(clone_engine_row)

        # [新增] VAD 预处理复选框
        self.chk_vad = QCheckBox("🔕 开启 VAD 预处理 (过滤环境底噪与静音)")
        self.chk_vad.setChecked(True)  # 默认开启
        self.chk_vad.setStyleSheet("font-size: 13px; padding: 5px;")
        self.chk_vad.setToolTip(
            "使用短时能量检测算法，智能压低非人声片段的背景噪音，\n"
            "提升 WORLD 声码器的转换纯净度和听感质量。"
        )
        layout.addWidget(self.chk_vad)

        # ===== [新增] VAD 参数调节区域：能量阈值 =====
        self.vad_title = QLabel("⚙️ VAD 参数微调")
        self.vad_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 10px 0px 5px 0px;")
        layout.addWidget(self.vad_title)

        # 能量阈值行
        vad_threshold_layout = QHBoxLayout()
        
        self.threshold_label = QLabel("能量阈值 (%):")
        self.threshold_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        
        self.label_aggressive = QLabel("激进降噪")
        self.label_aggressive.setStyleSheet("font-size: 11px; color: #ff6b6b; font-weight: bold;")
        
        self.slider_vad_threshold = QSlider(Qt.Horizontal)
        self.slider_vad_threshold.setRange(1, 15)  # 1% - 15%
        self.slider_vad_threshold.setValue(5)      # 默认 5%
        self.slider_vad_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_vad_threshold.setTickInterval(1)
        self.slider_vad_threshold.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #4CAF50;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        # 改成 QLineEdit 可输入百分比
        self.input_vad_threshold = QLineEdit("5")
        self.input_vad_threshold.setStyleSheet("""
            QLineEdit {
                font-size: 13px; 
                font-weight: bold;
                color: #4CAF50;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                min-width: 50px;
                max-width: 60px;
            }
        """)
        self.input_vad_threshold.setAlignment(Qt.AlignCenter)
        # 添加输入验证：只允许 1-15 的数字
        self.input_vad_threshold.textChanged.connect(self._on_threshold_input_changed)
        
        self.label_gentle = QLabel("温和降噪")
        self.label_gentle.setStyleSheet("font-size: 11px; color: #4ecdc4; font-weight: bold;")
        
        # 绑定滑块和输入框的同步
        self.slider_vad_threshold.valueChanged.connect(self._update_threshold_from_slider)
        
        vad_threshold_layout.addWidget(self.threshold_label)
        vad_threshold_layout.addWidget(self.label_aggressive)
        vad_threshold_layout.addWidget(self.slider_vad_threshold, 1)
        vad_threshold_layout.addWidget(self.input_vad_threshold)
        vad_threshold_layout.addWidget(self.label_gentle)
        layout.addLayout(vad_threshold_layout)

        # 帧长参数行
        vad_frame_layout = QHBoxLayout()
        
        self.frame_label = QLabel("帧长时间 (ms):")
        self.frame_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")
        
        self.label_short = QLabel("灵敏")
        self.label_short.setStyleSheet("font-size: 11px; color: #9b59b6; font-weight: bold;")
        
        self.slider_frame_ms = QSlider(Qt.Horizontal)
        self.slider_frame_ms.setRange(10, 50)     # 10ms - 50ms
        self.slider_frame_ms.setValue(20)         # 默认 20ms
        self.slider_frame_ms.setTickPosition(QSlider.TicksBelow)
        self.slider_frame_ms.setTickInterval(5)
        self.slider_frame_ms.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #9b59b6;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        self.input_frame_ms = QLineEdit("20")
        self.input_frame_ms.setStyleSheet("""
            QLineEdit {
                font-size: 13px; 
                font-weight: bold;
                color: #9b59b6;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                min-width: 50px;
                max-width: 60px;
            }
        """)
        self.input_frame_ms.setAlignment(Qt.AlignCenter)
        self.input_frame_ms.textChanged.connect(self._on_frame_input_changed)
        
        self.label_long = QLabel("稳定")
        self.label_long.setStyleSheet("font-size: 11px; color: #3498db; font-weight: bold;")
        
        self.slider_frame_ms.valueChanged.connect(self._update_frame_from_slider)
        
        vad_frame_layout.addWidget(self.frame_label)
        vad_frame_layout.addWidget(self.label_short)
        vad_frame_layout.addWidget(self.slider_frame_ms, 1)
        vad_frame_layout.addWidget(self.input_frame_ms)
        vad_frame_layout.addWidget(self.label_long)
        layout.addLayout(vad_frame_layout)

        # ===== 转换强度：控制 WORLD 合成与原声的混合比例 =====
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("转换强度:")
        self.strength_label.setStyleSheet("font-size:13px; font-weight:bold; color:#555; min-width:120px;")
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)
        self.slider_strength.setTickPosition(QSlider.TicksBelow)
        self.slider_strength.setTickInterval(10)
        self.input_strength = QLineEdit("100")
        self.input_strength.setStyleSheet("font-size:13px; color:#4CAF50; background-color:#f5f5f5;")
        self.input_strength.setMaximumWidth(60)
        self.input_strength.setAlignment(Qt.AlignCenter)
        # 同步
        self.slider_strength.valueChanged.connect(lambda v: self.input_strength.setText(str(v)))
        self.input_strength.textChanged.connect(lambda t: self._on_strength_input_changed(t))
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.slider_strength, 1)
        strength_layout.addWidget(self.input_strength)
        layout.addLayout(strength_layout)

        # 连接复选框信号到启用/禁用滑块和输入框的槽函数
        self.chk_vad.stateChanged.connect(self._on_vad_toggled)

        self.chk_phase2 = QCheckBox("🧼 开启第二阶段谱减噪 (noisereduce)")
        self.chk_phase2.setChecked(True)
        self.chk_phase2.setStyleSheet("font-size: 13px; padding: 5px;")
        self.chk_phase2.setToolTip(
            "在 VAD 后继续做谱减噪，进一步压制稳定背景噪声。\n"
            "可选温和 / 标准 / 强力 三档。"
        )
        layout.addWidget(self.chk_phase2)

        phase2_row = QHBoxLayout()
        self.phase2_label = QLabel("降噪模式:")
        self.phase2_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #555; min-width: 120px;")

        self.combo_phase2_mode = QComboBox()
        self.combo_phase2_mode.addItems(["温和", "标准", "强力"])
        self.combo_phase2_mode.setCurrentText("标准")
        self.combo_phase2_mode.setStyleSheet(
            "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; }"
        )
        self.combo_phase2_mode.setToolTip("温和保留更多原音，强力更适合背景噪声明显的录音。")

        self.chk_phase2.stateChanged.connect(self._on_phase2_toggled)

        phase2_row.addWidget(self.phase2_label)
        phase2_row.addWidget(self.combo_phase2_mode)
        phase2_row.addStretch(1)
        layout.addLayout(phase2_row)

        # 降噪策略：无 / 仅VAD / 仅谱减 / 两者
        strategy_row = QHBoxLayout()
        self.strat_label = QLabel("降噪策略:")
        self.strat_label.setStyleSheet("font-size:13px; font-weight:bold; color:#555; min-width:120px;")
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["无", "仅VAD", "仅谱减", "VAD->谱减"])
        self.combo_strategy.setCurrentText("VAD->谱减")
        self.combo_strategy.setStyleSheet("QComboBox { font-size:13px; padding:4px; min-width:180px; }")
        strategy_row.addWidget(self.strat_label)
        strategy_row.addWidget(self.combo_strategy)
        strategy_row.addStretch(1)
        layout.addLayout(strategy_row)

        # 播放时是否保留背景噪声（仅对播放转换后有效）
        self.chk_keep_noise = QCheckBox("播放转换后时保留背景噪声")
        self.chk_keep_noise.setChecked(False)
        self.chk_keep_noise.setToolTip("启用后播放转换后音频时会把原始录音中的背景噪声混回，便于A/B听感比较。")
        layout.addWidget(self.chk_keep_noise)

        self.btn_convert = QPushButton("开始转换")
        self.btn_convert.setMinimumHeight(38)
        self.btn_convert.setStyleSheet(
            "font-size: 15px; font-weight: bold; background-color: #4CAF50; color: white;"
        )
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        layout.addWidget(self.btn_convert)

        plot_row = QHBoxLayout()
        self.btn_plot_f0 = QPushButton("保存 F0 对比图")
        self.btn_plot_f0.setStyleSheet("font-size: 14px; background-color: #2196F3; color: white;")
        self.btn_plot_f0.clicked.connect(self.on_plot_f0_clicked)
        self.btn_plot_f0.setEnabled(False)
        plot_row.addWidget(self.btn_plot_f0)

        self.btn_plot_mel = QPushButton("保存 Mel 对比图")
        self.btn_plot_mel.setStyleSheet("font-size: 14px; background-color: #FF7043; color: white;")
        self.btn_plot_mel.clicked.connect(self.on_plot_mel_clicked)
        self.btn_plot_mel.setEnabled(False)
        plot_row.addWidget(self.btn_plot_mel)
        layout.addLayout(plot_row)

        player_row = QHBoxLayout()
        self.btn_play_original = QPushButton("播放原声")
        self.btn_play_original.clicked.connect(self.play_original)
        self.btn_play_original.setEnabled(False)
        player_row.addWidget(self.btn_play_original)

        self.btn_play_converted = QPushButton("播放转换后")
        self.btn_play_converted.clicked.connect(self.play_converted)
        self.btn_play_converted.setEnabled(False)
        player_row.addWidget(self.btn_play_converted)

        self.btn_stop = QPushButton("停止播放")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        player_row.addWidget(self.btn_stop)
        layout.addLayout(player_row)

        self.slider_position = QSlider(Qt.Horizontal)
        self.slider_position.setRange(0, 0)
        self.slider_position.sliderMoved.connect(self.seek_position)
        layout.addWidget(self.slider_position)

        self.label_time = QLabel("00:00 / 00:00")
        self.label_time.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label_time)

        wave_row = QHBoxLayout()
        self.label_wave = QLabel("实时振幅")
        self.wave_meter = QProgressBar()
        self.wave_meter.setRange(0, 100)
        self.wave_meter.setValue(0)
        self.wave_meter.setTextVisible(False)
        self.wave_meter.setMaximumWidth(110)
        self.wave_meter.setStyleSheet(
            "QProgressBar {border: 1px solid #666; border-radius: 4px; background: #1f1f1f;}"
            "QProgressBar::chunk {background-color: #00c853;}"
        )
        self.label_wave_value = QLabel("0%")
        self.label_wave_value.setMinimumWidth(40)
        self.label_wave_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.waveform_strip = WaveformStrip(self)

        wave_row.addWidget(self.label_wave)
        wave_row.addWidget(self.waveform_strip, 1)
        wave_row.addWidget(self.wave_meter)
        wave_row.addWidget(self.label_wave_value)
        layout.addLayout(wave_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: Consolas;")
        layout.addWidget(self.log_text)

        self._on_mode_changed()

    def load_file(self, filepath):
        # 加载新文件前，先停止任何播放
        if self.play_proc is not None:
            self.stop_playback()
        
        self.current_filepath = filepath
        self.converted_filepath = None

        self.drop_label.setText(f"\n\n源音频已加载:\n{os.path.basename(filepath)}\n（内部会自动转 WAV）\n\n")
        self.source_info_label.setText(f"源音频: {os.path.basename(filepath)}")
        self._report_audio_duration(filepath, "源音频")

        self.log_message(f"✅ 已准备好源文件: {filepath}")

        self._refresh_action_state()
        self.btn_plot_f0.setEnabled(True)
        self.btn_plot_mel.setEnabled(False)
        self.btn_play_original.setEnabled(True)
        self.btn_play_converted.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def load_target_file(self, filepath):
        # 单个目标文件
        self.target_filepaths = [filepath]
        self.target_drop_label.setText(f"\n\n目标音频已加载:\n{os.path.basename(filepath)}\n（内部会自动转 WAV）\n\n")
        self.target_info_label.setText(f"目标音频: {os.path.basename(filepath)}")
        self._report_audio_duration(filepath, "目标音频")
        self.log_message(f"✅ 已准备好目标文件: {filepath}")
        self._refresh_action_state()

    def load_target_files(self, file_list):
        # 多个目标文件
        self.target_filepaths = list(file_list)
        names = [os.path.basename(p) for p in self.target_filepaths]
        display = "\n\n目标音频已加载 (%d):\n%s\n（内部会自动转 WAV）\n\n" % (len(names), " | ".join(names))
        self.target_drop_label.setText(display)
        self.target_info_label.setText(f"目标音频: {len(names)} 个参考样本")
        for p in self.target_filepaths:
            self._report_audio_duration(p, "目标音频参考")
        self.log_message(f"✅ 已准备好 {len(names)} 个目标参考文件")
        self._refresh_action_state()

    def choose_source_file(self):
        # 打开文件选择对话框并加载为源音频
        try:
            exts = ' '.join(['*' + e for e in SUPPORTED_FORMATS])
            filter_str = f"音频文件 ({exts});;所有文件 (*)"
            path, _ = QFileDialog.getOpenFileName(self, "选择源音频", "", filter_str)
            if path:
                self.load_file(path)
        except Exception as e:
            self.log_message(f"⚠️ 打开源文件对话框失败: {e}")

    def choose_target_file(self):
        # 打开文件选择对话框并加载为目标音频
        try:
            exts = ' '.join(['*' + e for e in SUPPORTED_FORMATS])
            filter_str = f"音频文件 ({exts});;所有文件 (*)"
            paths, _ = QFileDialog.getOpenFileNames(self, "选择目标音频（可多选）", "", filter_str)
            if paths:
                if len(paths) == 1:
                    self.load_target_file(paths[0])
                else:
                    self.load_target_files(paths)
        except Exception as e:
            self.log_message(f"⚠️ 打开目标文件对话框失败: {e}")

    def log_message(self, msg):
        self.log_text.append(msg)

    def _on_mode_changed(self, *args):
        is_vc_mode = self.combo_mode.currentIndex() == 1
        self.target_title_label.setVisible(is_vc_mode)
        self.target_drop_label.setVisible(is_vc_mode)
        self.target_info_label.setVisible(is_vc_mode)

        world_controls = [
            self.vad_title,
            self.threshold_label,
            self.label_aggressive,
            self.label_gentle,
            self.chk_vad,
            self.slider_vad_threshold,
            self.input_vad_threshold,
            self.frame_label,
            self.label_short,
            self.label_long,
            self.slider_frame_ms,
            self.input_frame_ms,
            self.strength_label,
            self.slider_strength,
            self.input_strength,
            self.chk_phase2,
            self.phase2_label,
            self.combo_phase2_mode,
            self.strat_label,
            self.combo_strategy,
            self.chk_keep_noise,
        ]
        for widget in world_controls:
            widget.setVisible(not is_vc_mode)

        # 目标选择按钮也只在声线克隆模式可见
        try:
            self.btn_choose_target.setVisible(is_vc_mode)
        except Exception:
            pass
        
        try:
            self.clone_engine_label.setVisible(is_vc_mode)
            self.combo_clone_engine.setVisible(is_vc_mode)
        except Exception:
            pass

        self._refresh_action_state()

    def _refresh_action_state(self):
        if self.combo_mode.currentIndex() == 1:
            can_convert = bool(self.current_filepath and self.target_filepaths)
            self.btn_convert.setText("开始克隆")
            self.btn_plot_f0.setEnabled(bool(self.current_filepath))
            self.btn_plot_mel.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
        else:
            can_convert = bool(self.current_filepath)
            self.btn_convert.setText("开始转换")
            self.btn_plot_f0.setEnabled(bool(self.current_filepath))
            self.btn_plot_mel.setEnabled(bool(self.converted_filepath and os.path.exists(self.converted_filepath)))
        self.btn_convert.setEnabled(can_convert)

    def _report_audio_duration(self, filepath, role):
        try:
            duration, warnings = _duration_status(filepath, role)
            self.log_message(f"ℹ️ {role}时长: {duration:.2f}s")
            show_warnings = role.startswith("目标")
            if show_warnings:
                show_popup = self.combo_mode.currentIndex() == 1
                for warn in warnings:
                    self.log_message(f"⚠️ {warn}")
                    if show_popup:
                        QMessageBox.warning(self, "时长提醒", warn)
            return duration
        except Exception as e:
            self.log_message(f"⚠️ 无法检测{role}时长: {e}")
            return None

    def _update_threshold_from_slider(self):
        """滑块变化时更新输入框"""
        value = self.slider_vad_threshold.value()
        self.input_vad_threshold.blockSignals(True)
        self.input_vad_threshold.setText(str(value))
        self.input_vad_threshold.blockSignals(False)

    def _on_threshold_input_changed(self, text):
        """输入框变化时同步滑块"""
        try:
            value = int(text)
            if 1 <= value <= 15:
                self.slider_vad_threshold.blockSignals(True)
                self.slider_vad_threshold.setValue(value)
                self.slider_vad_threshold.blockSignals(False)
            else:
                # 超出范围时不更新滑块
                pass
        except ValueError:
            # 非数字输入，忽略
            pass

    def _update_frame_from_slider(self):
        """帧长滑块变化时更新输入框"""
        value = self.slider_frame_ms.value()
        self.input_frame_ms.blockSignals(True)
        self.input_frame_ms.setText(str(value))
        self.input_frame_ms.blockSignals(False)

    def _on_frame_input_changed(self, text):
        """帧长输入框变化时同步滑块"""
        try:
            value = int(text)
            if 10 <= value <= 50:
                self.slider_frame_ms.blockSignals(True)
                self.slider_frame_ms.setValue(value)
                self.slider_frame_ms.blockSignals(False)
            else:
                # 超出范围时不更新滑块
                pass
        except ValueError:
            # 非数字输入，忽略
            pass

    def _on_strength_input_changed(self, text):
        try:
            value = int(text)
            if 0 <= value <= 100:
                self.slider_strength.blockSignals(True)
                self.slider_strength.setValue(value)
                self.slider_strength.blockSignals(False)
            else:
                pass
        except ValueError:
            pass

    def _on_vad_toggled(self):
        """当 VAD 复选框状态改变时，启用/禁用滑块和输入框"""
        is_checked = self.chk_vad.isChecked()
        self.slider_vad_threshold.setEnabled(is_checked)
        self.input_vad_threshold.setEnabled(is_checked)
        self.slider_frame_ms.setEnabled(is_checked)
        self.input_frame_ms.setEnabled(is_checked)
        
        # 如果禁用 VAD，所有控件变灰
        if not is_checked:
            gray_style_slider = """
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #aaa;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """
            self.slider_vad_threshold.setStyleSheet(gray_style_slider)
            self.slider_frame_ms.setStyleSheet(gray_style_slider)
            
            gray_input_style = """
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #aaa;
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """
            self.input_vad_threshold.setStyleSheet(gray_input_style)
            self.input_frame_ms.setStyleSheet(gray_input_style)
        else:
            # 恢复原来的样式
            self.slider_vad_threshold.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #4CAF50;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """)
            self.slider_frame_ms.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #ddd;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #9b59b6;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            """)
            self.input_vad_threshold.setStyleSheet("""
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #4CAF50;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """)
            self.input_frame_ms.setStyleSheet("""
                QLineEdit {
                    font-size: 13px; 
                    font-weight: bold;
                    color: #9b59b6;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 4px;
                    min-width: 50px;
                    max-width: 60px;
                }
            """)

    def _on_phase2_toggled(self):
        is_checked = self.chk_phase2.isChecked()
        self.combo_phase2_mode.setEnabled(is_checked)
        if not is_checked:
            self.combo_phase2_mode.setStyleSheet(
                "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; color: #888; }"
            )
        else:
            self.combo_phase2_mode.setStyleSheet(
                "QComboBox { font-size: 13px; font-weight: bold; padding: 4px; min-width: 140px; }"
            )


    def start_conversion(self):
        if not self.current_filepath:
            return

        is_vc_mode = self.combo_mode.currentIndex() == 1
        if is_vc_mode and not self.target_filepaths:
            QMessageBox.warning(self, "缺少目标音频", "声线克隆模式需要同时选择目标音频。")
            self.log_message("⚠️ 声线克隆模式缺少目标音频。")
            return

        # 在开始转换前，先停止任何正在进行的播放
        if self.play_proc is not None:
            self.log_message("⏹️ 停止现有播放以开始转换...")
            self.stop_playback()

        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("正在处理中...")

        # 时长检查：克隆模式过短直接提醒并停止；单音模式仅提示
        try:
            source_duration, source_warnings = _duration_status(self.current_filepath, "源音频")
            # 源音频不再做时长限制与提醒
        except Exception as e:
            self.log_message(f"⚠️ 源音频时长检测失败: {e}")
            QMessageBox.warning(self, "时长检测失败", f"源音频时长检测失败：{e}")
            self.btn_convert.setEnabled(True)
            self._refresh_action_state()
            return

        if is_vc_mode:
            try:
                for idx, target_path in enumerate(self.target_filepaths, start=1):
                    target_duration, target_warnings = _duration_status(target_path, f"目标音频{idx}")
                    if target_duration < MIN_AUDIO_LENGTH:
                        warn_text = target_warnings[0] if target_warnings else f"目标音频{idx}太短，至少需要 {MIN_AUDIO_LENGTH:.0f}s。"
                        QMessageBox.warning(self, "时长不足", warn_text)
                        self.log_message(f"⚠️ {warn_text}")
                        self.btn_convert.setEnabled(True)
                        self._refresh_action_state()
                        return
                    for warn_text in target_warnings:
                        self.log_message(f"⚠️ {warn_text}")
                        QMessageBox.warning(self, "时长提醒", warn_text)
            except Exception as e:
                self.log_message(f"⚠️ 目标音频时长检测失败: {e}")
                QMessageBox.warning(self, "时长检测失败", f"目标音频时长检测失败：{e}")
                self.btn_convert.setEnabled(True)
                self._refresh_action_state()
                return

            engine = "FreeVC"
            try:
                engine = self.combo_clone_engine.currentText()
            except Exception:
                pass

            self.thread = VoiceCloneThread(self.current_filepath, self.target_filepaths, engine=engine)
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.conversion_finished)
            self.thread.start()
            return

        # WORLD 单音频模式：继续使用原有 VAD / 谱减噪流程
        try:
            threshold_value = int(self.input_vad_threshold.text())
            threshold_ratio = max(1, min(15, threshold_value)) / 100.0
        except ValueError:
            threshold_ratio = 0.05

        try:
            frame_value = int(self.input_frame_ms.text())
            frame_ms = max(10, min(50, frame_value))
        except ValueError:
            frame_ms = 20

        strat = self.combo_strategy.currentText().strip()
        if strat == "无":
            use_vad = False
            use_phase2 = False
        elif strat == "仅VAD":
            use_vad = True
            use_phase2 = False
        elif strat == "仅谱减":
            use_vad = False
            use_phase2 = True
        else:
            use_vad = True
            use_phase2 = True

        phase2_mode = self.combo_phase2_mode.currentText().strip()

        self.thread = VoiceConverterThread(
            self.current_filepath,
            enable_vad=use_vad,
            threshold_ratio=threshold_ratio,
            frame_ms=frame_ms,
            enable_phase2=use_phase2,
            phase2_mode=phase2_mode,
            strength=max(0.0, min(1.0, float(self.slider_strength.value()) / 100.0)),
        )
        self.thread.log_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.conversion_finished)
        self.thread.start()

    def conversion_finished(self, output_path):
        self._refresh_action_state()
        if output_path:
            self.converted_filepath = output_path
            self.log_message(f"文件已保存至: {output_path}")
            self.log_message("-" * 32)
            self.btn_play_converted.setEnabled(True)
            self.btn_plot_mel.setEnabled(True)
        else:
            self.log_message("转换失败。")

    def on_plot_f0_clicked(self):
        if not self.current_filepath:
            return
        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_f0_hist.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 F0 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_f0.setEnabled(False)
        self.log_message(f"正在保存 F0 对比图: {save_path}")
        self.plot_f0_thread = PlotF0Thread(self.current_filepath, save_path)
        self.plot_f0_thread.log_signal.connect(self.log_message)
        self.plot_f0_thread.finished_signal.connect(self.plot_f0_finished)
        self.plot_f0_thread.start()

    def plot_f0_finished(self, path):
        self.btn_plot_f0.setEnabled(True)
        if path:
            self.log_message(f"F0 图像已保存: {path}")
        else:
            self.log_message("F0 绘图未完成或失败。")

    def on_plot_mel_clicked(self):
        if not self.current_filepath:
            return
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("请先完成转换，再生成 Mel 对比图。")
            return

        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_mel_compare.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Mel 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_mel.setEnabled(False)
        self.log_message(f"正在保存 Mel 对比图: {save_path}")
        self.plot_mel_thread = PlotMelThread(self.current_filepath, self.converted_filepath, save_path)
        self.plot_mel_thread.log_signal.connect(self.log_message)
        self.plot_mel_thread.finished_signal.connect(self.plot_mel_finished)
        self.plot_mel_thread.start()

    def plot_mel_finished(self, path):
        self.btn_plot_mel.setEnabled(True)
        if path:
            self.log_message(f"Mel 图像已保存: {path}")
        else:
            self.log_message("Mel 绘图未完成或失败。")

    def _play_file(self, file_path, tag):
        if not file_path or not os.path.exists(file_path):
            self.log_message(f"❌ {tag}文件不存在，无法播放。")
            return

        self.current_play_path = file_path
        
        # 显示文件信息
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.log_message(f"📂 准备播放 {tag}: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
        
        play_path = file_path
        # 如果不是 WAV，先转换为临时 16k WAV 以兼容 playback_worker / soundfile
        try:
            if not file_path.lower().endswith('.wav'):
                tmp = _convert_to_temp_wav(file_path, prefix='play_tmp_')
                play_path = tmp
                self._temp_play_path = tmp
        except Exception as e:
            self.log_message(f"⚠️ 无法为播放生成临时 WAV，尝试直接播放原文件: {e}")

        ok = self._start_playback(play_path, 0)
        if ok:
            self.log_message(f"▶️ 正在播放 {tag}: {os.path.basename(file_path)}")
        else:
            self.log_message(f"❌ 播放启动失败。")

    def play_original(self):
        self._play_file(self.current_filepath, "原声")

    def play_converted(self):
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("❌ 转换后文件不存在，无法播放。")
            return

        play_path = self.converted_filepath
        # 如果用户选择播放时保留背景噪声，生成临时混合文件
        if self.chk_keep_noise.isChecked():
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                tmp_path = tmp.name
                tmp.close()

                # 读取 converted 和 原始音频
                y_conv, sr_conv = sf.read(self.converted_filepath, dtype='float64')
                x_orig, sr_orig = sf.read(self.current_filepath, dtype='float64')
                if sr_conv != sr_orig:
                    # 重采样原始到 converted 的采样率
                    x_orig = librosa.resample(x_orig, orig_sr=sr_orig, target_sr=sr_conv)
                    sr = sr_conv
                else:
                    sr = sr_conv

                # 载入 mask（优先使用 sidecar），否则生成一个简单 VAD mask
                mask_path = self.converted_filepath + ".mask.npy"
                mask = None
                if os.path.exists(mask_path):
                    try:
                        mask = np.load(mask_path)
                    except Exception:
                        mask = None

                if mask is None:
                    try:
                        _, mask = apply_energy_vad(x_orig, sr, frame_ms=int(self.input_frame_ms.text()), threshold_ratio=max(1,min(15,int(self.input_vad_threshold.text())))/100.0)
                    except Exception:
                        mask = np.ones_like(x_orig)

                # 调整长度
                minlen = min(len(y_conv), len(x_orig), len(mask))
                y_conv = np.asarray(y_conv[:minlen], dtype=np.float64)
                x_orig = np.asarray(x_orig[:minlen], dtype=np.float64)
                mask = np.asarray(mask[:minlen], dtype=np.float64)

                # 混合：语音部分使用转换后的，噪声部分从原始音频取一部分（gain 0.3）
                background_gain = 0.35
                mixed = y_conv * mask + x_orig * (1.0 - mask) * background_gain
                # 防止溢出
                peak = np.max(np.abs(mixed))
                if peak > 1.0:
                    mixed = mixed / peak * 0.95

                sf.write(tmp_path, mixed, int(sr))
                play_path = tmp_path
                # 记录临时文件用于后续清理
                self._temp_play_path = tmp_path
            except Exception as e:
                self.log_message(f"⚠️ 生成带背景混合文件失败，使用原始转换文件播放: {e}")
                play_path = self.converted_filepath

        self._play_file(play_path, "转换后")

    def stop_playback(self):
        """完全停止播放并清理资源"""
        # 终止播放进程
        if self.play_proc is not None:
            try:
                # 强制终止进程
                self.play_proc.terminate()
                # 等待进程完全退出（最多 500ms）
                try:
                    self.play_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.play_proc.kill()
                    self.play_proc.wait()
            except Exception as e:
                self.log_message(f"⚠️ 清理播放进程时出错: {e}")
            finally:
                self.play_proc = None
        
        # 停止定时器
        self.play_timer.stop()
        
        # 清空播放相关的缓存
        self.play_data = None
        self.play_obj = None
        
        # 重置波形显示
        self._reset_wave_meter()
        
        self.log_message("⏹️ 播放已停止。")

        # 删除临时混合文件（如果有）
        try:
            if hasattr(self, '_temp_play_path') and self._temp_play_path:
                if os.path.exists(self._temp_play_path):
                    os.remove(self._temp_play_path)
                self._temp_play_path = None
        except Exception:
            pass

    def seek_position(self, value_ms):
        if not self.current_play_path:
            return
        ok = self._start_playback(self.current_play_path, value_ms)
        if not ok:
            self.log_message("拖动定位失败，已停止播放。")

    def _start_playback(self, file_path, offset_ms):
        """启动音频播放，offset_ms 为从文件开始的偏移量（毫秒）"""
        # 首先完全停止任何正在进行的播放
        if self.play_proc is not None:
            try:
                self.play_proc.terminate()
                try:
                    self.play_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.play_proc.kill()
                    self.play_proc.wait()
            except Exception:
                pass
            self.play_proc = None
        
        try:
            # 读取音频文件进行时间计算
            data, rate = sf.read(file_path, dtype="int16")
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]

            # simpleaudio 对通道和内存布局比较敏感，先规整数据。
            if channels > 2:
                data = data[:, :2]
                channels = 2

            data = np.ascontiguousarray(data, dtype=np.int16)

            # 验证偏移量
            start_sample = int(max(offset_ms, 0) * rate / 1000)
            if start_sample >= len(data):
                self.log_message("⚠️ 播放位置超出音频长度。")
                return False

            # 存储播放状态
            self.play_data = data
            self.play_rate = int(rate)
            self.play_channels = channels
            self.play_total_ms = int(len(data) * 1000 / rate)
            self.play_start_ms = int(offset_ms)
            self.play_start_epoch = time.time()

            self.slider_position.setRange(0, self.play_total_ms)

            # 获取 playback_worker.py 的路径
            worker_path = os.path.join(os.path.dirname(__file__), "playback_worker.py")
            if not os.path.exists(worker_path):
                self.log_message("❌ 播放脚本 playback_worker.py 不存在。")
                return False

            # Windows 下创建隐藏进程
            creation_flags = 0
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            # 启动播放进程（subprocess 隔离）
            self.play_proc = subprocess.Popen(
                [sys.executable, worker_path, file_path, str(int(offset_ms))],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
            
            # 启动定时器用于进度追踪
            self.play_timer.start()
            return True
            
        except Exception as e:
            # 播放失败，清理资源
            self.play_proc = None
            self.play_timer.stop()
            self._reset_wave_meter()
            self.log_message(f"❌ 播放异常: {e}")
            return False

    def _on_playback_timer(self):
        """处理播放进度更新和进程监控"""
        if self.play_proc is None:
            self.play_timer.stop()
            self._reset_wave_meter()
            return

        # 计算当前播放位置
        elapsed_ms = int((time.time() - self.play_start_epoch) * 1000)
        current_ms = min(self.play_start_ms + elapsed_ms, self.play_total_ms)

        # 更新进度条
        self.slider_position.blockSignals(True)
        self.slider_position.setValue(current_ms)
        self.slider_position.blockSignals(False)
        
        # 更新时间显示
        self.label_time.setText(f"{self._fmt_ms(current_ms)} / {self._fmt_ms(self.play_total_ms)}")
        
        # 更新波形显示
        self._update_wave_meter(current_ms)

        # 检查播放进程状态
        rc = self.play_proc.poll()
        if rc is not None:
            # 进程已退出
            self.play_timer.stop()
            
            # 读取进程输出用于调试
            try:
                stdout, stderr = self.play_proc.communicate(timeout=0.1)
                if stderr:
                    error_msg = stderr.decode('utf-8', errors='ignore').strip()
                    if error_msg:
                        self.log_message(f"🔧 [播放进程] {error_msg}")
            except:
                pass
            
            self.play_proc = None
            self._reset_wave_meter()
            # 清理临时混合文件（如果存在）
            try:
                if hasattr(self, '_temp_play_path') and self._temp_play_path:
                    if os.path.exists(self._temp_play_path):
                        os.remove(self._temp_play_path)
                    self._temp_play_path = None
            except Exception:
                pass
            
            if rc != 0:
                # 进程异常退出
                if rc == 3221225477:
                    # 特殊处理这个特定的崩溃代码
                    self.log_message(f"⚠️ 播放进程崩溃 (返回码: {rc}) - 可能是音频格式问题或内存错误")
                else:
                    self.log_message(f"⚠️ 播放进程异常退出，返回码: {rc}")

    def _update_wave_meter(self, current_ms):
        if self.play_data is None or self.play_rate <= 0:
            self._reset_wave_meter()
            return

        sample_idx = int(current_ms * self.play_rate / 1000)
        window_size = max(int(self.play_rate * 0.025), 1)
        left = max(sample_idx - window_size // 2, 0)
        right = min(sample_idx + window_size // 2, len(self.play_data))
        if right <= left:
            self._reset_wave_meter()
            return

        segment = self.play_data[left:right]
        segment = np.asarray(segment, dtype=np.float32)
        if segment.ndim > 1:
            segment = np.max(np.abs(segment), axis=1)
        else:
            segment = np.abs(segment)

        peak = float(np.max(segment)) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(segment))) / 32768.0)
        level = min(1.0, 0.55 * (peak ** 0.5) + 0.45 * (rms ** 0.5))
        meter_value = int(level * 100.0)
        self.wave_meter.setValue(meter_value)
        self.label_wave_value.setText(f"{meter_value}%")
        self.waveform_strip.push_level(level)

    def _reset_wave_meter(self):
        self.wave_meter.setValue(0)
        self.label_wave_value.setText("0%")
        self.waveform_strip.reset()

    @staticmethod
    def _fmt_ms(ms):
        seconds = int(ms / 1000)
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VoiceChangerApp()
    ex.show()
    sys.exit(app.exec_())
