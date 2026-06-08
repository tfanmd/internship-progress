import speech_recognition as sr
import torch
import numpy as np
import io
import pygame
import queue
import time
from gtts import gTTS
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

# ==========================================
# 1. PENGATURAN PERANGKAT DAN MODEL
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan perangkat: {device.upper()}")

print("Memuat model ASR dan MT...")
model_id_asr = "openai/whisper-small"
processor_asr = AutoProcessor.from_pretrained(model_id_asr)
model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_asr).to(device)

model_id_mt = "Helsinki-NLP/opus-mt-en-id"
tokenizer_mt = MarianTokenizer.from_pretrained(model_id_mt)
model_mt = AutoModelForSeq2SeqLM.from_pretrained(model_id_mt).to(device)

pygame.mixer.init()
print("Semua model siap digunakan.\n")

# Membuat antrean (queue) untuk menampung potongan suara dari mikrofon
antrean_suara = queue.Queue()

# ==========================================
# 2. FUNGSI PEKERJA 1 (TELINGA / MIKROFON Latar Belakang)
# ==========================================
def penangkap_suara(recognizer, audio):
    # Fungsi ini otomatis dipanggil setiap kali pengguna menjeda ucapan.
    # Memasukkan audio yang terekam ke dalam antrean untuk diproses.
    antrean_suara.put(audio)

# ==========================================
# 3. FUNGSI PEKERJA 2 (OTAK & MULUT / PEMROSESAN UTAMA)
# ==========================================
def jalankan_sistem_simultan():
    r = sr.Recognizer()
    r.pause_threshold = 1.0 # Jeda 1 detik sebagai pemotong kalimat

    r.non_speaking_duration = 0.5 # Jeda 0.5 detik untuk memastikan benar-benar selesai bicara
    
    mikrofon = sr.Microphone(sample_rate=16000)
    
    with mikrofon as source:
        print("Menyesuaikan dengan kebisingan ruangan... (Mohon diam sejenak)")
        r.adjust_for_ambient_noise(source, duration=2.0)
    
    print("\n" + "="*50)
    print("🎧 SISTEM TERJEMAHAN SIMULTAN AKTIF")
    print("Mikrofon akan terus mendengarkan di latar belakang.")
    print("Silakan bicara dengan jeda alami untuk memicu terjemahan.")
    print("="*50 + "\n")
    
    # Memulai pendengaran di latar belakang (Tidak memblokir proses selanjutnya)
    stop_listening = r.listen_in_background(mikrofon, penangkap_suara)
    
    try:
        while True:
            # Mengambil potongan suara dari antrean (Proses menunggu di sini jika antrean kosong)
            audio_masuk = antrean_suara.get()
            
            # Konversi audio ke format Numpy (16kHz, float32)
            wav_bytes = audio_masuk.get_wav_data(convert_rate=16000, convert_width=2)
            audio_array = np.frombuffer(wav_bytes, dtype=np.int16)
            data_suara = audio_array.astype(np.float32) / 32768.0 
            
            # TAHAP 1: Transkripsi (ASR)
            input_features = processor_asr(
                data_suara, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)

            forced_decoder_ids = processor_asr.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model_asr.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            teks_inggris = processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            # Mengabaikan keheningan atau suara tidak valid
            if not teks_inggris or teks_inggris.lower() in ["you", "thank you", "thanks", "you."]:
                continue
                
            print(f"📝 Anda : {teks_inggris}")
            
            # TAHAP 2: Terjemahan (MT)
            inputs_mt = tokenizer_mt(teks_inggris, return_tensors="pt", padding=True).to(device)
            translated_ids = model_mt.generate(**inputs_mt, max_length=512)
            teks_indo = tokenizer_mt.batch_decode(translated_ids, skip_special_tokens=True)[0].strip()
            
            print(f"🌐 AI   : {teks_indo}")
            
            # TAHAP 3: Membuat dan Memutar Suara (TTS RAM-Only)
            memori_suara = io.BytesIO()
            gTTS(teks_indo, lang="id").write_to_fp(memori_suara)
            memori_suara.seek(0)
            
            pygame.mixer.music.load(memori_suara)
            pygame.mixer.music.play()
            
            # Menunggu suara AI selesai agar tidak menumpuk dengan terjemahan berikutnya
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            if hasattr(pygame.mixer.music, 'unload'):
                pygame.mixer.music.unload()
            
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh pengguna.")
        # Menghentikan pendengaran latar belakang secara aman
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    jalankan_sistem_simultan()