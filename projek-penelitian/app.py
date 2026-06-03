import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from transformers import MarianTokenizer
import pygame
import time
import os
import sounddevice as sd
import soundfile as sf

# 1. SETUP HARDWARE (Otomatis deteksi GPU / CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan hardware: {device.upper()}")

# ==========================================
# 2. PROSES LOADING MODEL (Telinga & Otak)
# ==========================================
print("\nSedang memuat model ke RAM/VRAM (Pastikan internet nyala)...")

# ASR - Whisper Tiny
model_id_asr = "openai/whisper-tiny" # Pakai tiny biar enteng di lokal
processor_asr = AutoProcessor.from_pretrained(model_id_asr)
model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_asr).to(device)

# MT - Helsinki
model_id_mt = "Helsinki-NLP/opus-mt-en-id"
# tokenizer_mt = AutoTokenizer.from_pretrained(model_id_mt)
tokenizer_mt = MarianTokenizer.from_pretrained(model_id_mt)
model_mt = AutoModelForSeq2SeqLM.from_pretrained(model_id_mt).to(device)

print("Semua model siap tempur!\n")

def rekam_dari_mic(durasi=7, file_output="input_live.wav", fs=16000):
    print("\n" + "="*40)
    input("👉 Tekan [ENTER] untuk mulai ngomong bahasa Inggris...")
    
    print(f"🔴 Merekam selama {durasi} detik! Ngomong sekarang cuy...")
    # channels=1 artinya merekam dalam format Mono (sesuai standar Whisper)
    rekaman = sd.rec(int(durasi * fs), samplerate=fs, channels=1)
    
    # Tunggu sampai durasi habis
    sd.wait() 
    
    # Simpan suara ke file sementara
    sf.write(file_output, rekaman, fs)
    print("✅ Perekaman selesai!")
    print("="*40 + "\n")
    
    return file_output
# 3. FUNGSI UTAMA PIPELINE E2E
def smart_headset(file_audio_masuk):
    if not os.path.exists(file_audio_masuk):
        print(f"❌ Error: File {file_audio_masuk} nggak ketemu di folder ini!")
        return

    # --- TAHAP 1: ASR ---
    print("🎙️ TAHAP 1: Mendengarkan suara...")
    start_asr = time.time() # ⏱️ Mulai Stopwatch ASR
    
    audio_array, sampling_rate = librosa.load(file_audio_masuk, sr=16000)
    input_features = processor_asr(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
    predicted_ids = model_asr.generate(input_features)
    teks_transkrip = processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    end_asr = time.time() # ⏱️ Stop Stopwatch ASR
    waktu_asr = end_asr - start_asr
    print(f"   📝 Transkrip : '{teks_transkrip}'")
    print(f"   ⏱️ Waktu ASR : {waktu_asr:.2f} detik")

    # ==========================================
    # --- TAHAP 2: MT (Dengan Chunking) ---
    # ==========================================
    print("\n🧠 TAHAP 2: Menerjemahkan makna...")
    start_mt = time.time() # ⏱️ Mulai Stopwatch MT
    
    daftar_kalimat = [k.strip() + "." for k in teks_transkrip.split('.') if k.strip()]
    hasil_terjemahan_lengkap = []
    
    for kalimat in daftar_kalimat:
        inputs_mt = tokenizer_mt(kalimat, return_tensors="pt", padding=True).to(device)
        translated_ids = model_mt.generate(**inputs_mt, max_length=512)
        teks_indo_potongan = tokenizer_mt.batch_decode(translated_ids, skip_special_tokens=True)[0].strip()
        hasil_terjemahan_lengkap.append(teks_indo_potongan)
    
    teks_indo = " ".join(hasil_terjemahan_lengkap)
    
    end_mt = time.time() # ⏱️ Stop Stopwatch MT
    waktu_mt = end_mt - start_mt
    print(f"   🌐 Terjemahan: '{teks_indo}'")
    print(f"   ⏱️ Waktu MT  : {waktu_mt:.2f} detik")

    # --- TAHAP 3: TTS ---
    print("\n🔊 TAHAP 3: Membuat suara balasan...")
    start_tts = time.time() # ⏱️ Mulai Stopwatch TTS
    
    file_output = "output_suara_lokal.mp3"
    gTTS(teks_indo, lang="id").save(file_output)
    
    end_tts = time.time() # ⏱️ Stop Stopwatch TTS
    waktu_tts = end_tts - start_tts
    print(f"   ⏱️ Waktu TTS : {waktu_tts:.2f} detik")
    
    # ==========================================
    # --- TAHAP 4: PEMUTARAN AUDIO ---
    # ==========================================
    print("\n▶️ Memutar audio di speaker...")
    pygame.mixer.init()
    pygame.mixer.music.load(file_output)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    
    waktu_total = waktu_asr + waktu_mt + waktu_tts
    print(f"\n✅ SIMULASI SELESAI! (Total Latency Sistem: {waktu_total:.2f} detik)")


# ==========================================
# 4. EKSEKUSI
# ==========================================
# Pastikan lu punya file audio bahasa inggris (misal: 'tes_inggris.wav') di satu folder dengan script ini
# file_audio = "audio_eng_2487.wav" 

# print("🚀 MEMULAI SISTEM SMART HEADSET...")

if __name__ == "__main__":
    print("🚀 MEMULAI SISTEM SMART HEADSET (LIVE MIC MODE)...")
    
    # 1. Panggil fungsi rekam (misal kita set 7 detik)
    file_suara_langsung = rekam_dari_mic(durasi=7)
    
    # 2. Masukkan hasil rekaman mic langsung ke otak AI
    smart_headset(file_suara_langsung)
# Buka comment di bawah ini kalau lu udah nyiapin file suaranya:
# smart_headset(file_audio)