import os
import json

import torch
import numpy as np
from scipy.signal import resample_poly
from math import gcd

import hifigan
from model import FastSpeech2, ScheduledOptim


# ---------------------------------------------------------------------------
# Vocoder sampling-rate bridge
# ---------------------------------------------------------------------------
# The bundled HiFi-GAN universal checkpoint was trained on 22,050 Hz audio.
# FastSpeech 2 was trained on 16,000 Hz audio.  The two are incompatible in
# terms of absolute sample rate.  Because HiFi-GAN upsamples by a fixed
# factor of 256 (rates=[8,8,2,2]), its waveform output represents speech at
# 22,050 Hz regardless of the frame-rate of the input mel-spectrogram.
# Playing that output at 16,000 Hz would pitch speech down 5.5 semitones.
#
# FIX: detect the mismatch and resample (22,050 → 16,000) so that the saved
# WAV files are at the correct 16 kHz rate with correct pitch and tempo.
# This is an interim fix; the long-term solution is a native 16 kHz HiFi-GAN.

_HIFIGAN_NATIVE_SR = 22050   # sample rate the vocoder was trained at


def _get_vocoder_native_sr(hifigan_dir):
    """Return the sample rate declared in hifigan/config.json."""
    cfg_path = os.path.join(hifigan_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return int(cfg.get("sampling_rate", _HIFIGAN_NATIVE_SR))
    return _HIFIGAN_NATIVE_SR


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    restore_step = getattr(args, "restore_step", 0)
    if restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(
                train_config["path"]["ckpt_path"],
                "step_{}.pth".format(restore_step),
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hifigan_dir = os.path.join(pkg_root, "hifigan")
        cfg_path = os.path.join(hifigan_dir, "config.json")
        if not os.path.exists(cfg_path):
            cfg_path = "hifigan/config.json"
            hifigan_dir = "hifigan"
        with open(cfg_path, "r") as f:
            hifigan_cfg_dict = json.load(f)
        hifigan_cfg = hifigan.AttrDict(hifigan_cfg_dict)
        vocoder = hifigan.Generator(hifigan_cfg)
        if speaker == "LJSpeech":
            ckpt_path = os.path.join(hifigan_dir, "generator_LJSpeech.pth.tar")
            if not os.path.exists(ckpt_path):
                ckpt_path = "hifigan/generator_LJSpeech.pth.tar"
        elif speaker == "universal":
            ckpt_path = os.path.join(hifigan_dir, "generator_universal.pth.tar")
            if not os.path.exists(ckpt_path):
                ckpt_path = "hifigan/generator_universal.pth.tar"
        ckpt = torch.load(ckpt_path, map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
        # Store native SR alongside the vocoder for resampling in vocoder_infer
        vocoder._native_sr = int(hifigan_cfg_dict.get("sampling_rate", _HIFIGAN_NATIVE_SR))

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    target_sr = int(preprocess_config["preprocessing"]["audio"]["sampling_rate"])
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs_np = wavs.cpu().numpy()

    # ---------------------------------------------------------------------------
    # Vocoder sample-rate bridge (FIX: CRITICAL vocoder incompatibility)
    # ---------------------------------------------------------------------------
    # HiFi-GAN universal was trained on 22,050 Hz audio. FastSpeech 2 was
    # trained on 16,000 Hz audio. The vocoder generates waveforms whose sample
    # rate matches the vocoder's training sample rate (22,050 Hz).  We resample
    # to the acoustic model's target sample rate before writing/returning.
    #
    # Without this fix: playing 22,050 Hz waveform at 16,000 Hz drops all
    # frequencies by factor 16000/22050 = 0.7256 (~5.5 semitones downward).
    vocoder_native_sr = getattr(vocoder, "_native_sr", _HIFIGAN_NATIVE_SR)
    if name == "HiFi-GAN" and vocoder_native_sr != target_sr:
        # Compute irreducible fraction for polyphase resampling
        common = gcd(target_sr, vocoder_native_sr)
        up = target_sr // common
        down = vocoder_native_sr // common
        resampled = []
        for wav in wavs_np:
            wav_f = wav.astype(np.float64)
            wav_r = resample_poly(wav_f, up, down)
            resampled.append(wav_r)
        wavs_np = resampled
        # lengths are in samples at vocoder_native_sr; rescale to target_sr
        if lengths is not None:
            lengths = [int(l * target_sr / vocoder_native_sr) for l in lengths]
    else:
        wavs_np = list(wavs_np)

    # Convert to int16
    result = []
    for i, wav in enumerate(wavs_np):
        wav_int16 = (wav * max_wav_value).clip(-max_wav_value, max_wav_value).astype("int16")
        if lengths is not None:
            wav_int16 = wav_int16[: lengths[i]]
        result.append(wav_int16)

    return result
