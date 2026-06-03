import gradio as gr
import torch
import librosa
import time
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from transformers import MarianTokenizer
from gtts import gTTS
import soundfile as sf
import numpy as np

# ==========================================
# 1. SETUP HARDWARE (Otomatis deteksi GPU / CPU)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Menggunakan hardware: {device.upper()}")

# ==========================================
# 2. PROSES LOADING MODEL (Telinga & Otak)
# ==========================================
print("\n⏳ Sedang memuat model ke RAM/VRAM (Pastikan internet nyala)...")

# ASR - Whisper Tiny
model_id_asr = "openai/whisper-tiny" 
processor_asr = AutoProcessor.from_pretrained(model_id_asr)
model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_asr).to(device)

# MT - Helsinki
model_id_mt = "Helsinki-NLP/opus-mt-en-id"
tokenizer_mt = MarianTokenizer.from_pretrained(model_id_mt)
model_mt = AutoModelForSeq2SeqLM.from_pretrained(model_id_mt).to(device)

print("✅ Semua model siap tempur!\n")

# ==========================================
# 3. FUNGSI UTAMA (Diubah pakai 'return' buat Web)
# ==========================================
def smart_headset_web(audio_masuk):
    if audio_masuk is None:
        return "❌ Tolong rekam suara dulu cuy!", "Tidak ada terjemahan.", None

    try:
        print("\n" + "="*40)
        print("🎙️ TAHAP 1: Memproses audio murni di RAM...")
        start_asr = time.time() 
        
        # 1. Bongkar data dari Gradio
        sr_asli, data_suara = audio_masuk
        
        # 2. Normalisasi Tipe Data (Wajib biar Whisper nggak halusinasi "You")
        if data_suara.dtype == np.int16:
            data_suara = data_suara.astype(np.float32) / 32768.0
        elif data_suara.dtype == np.int32:
            data_suara = data_suara.astype(np.float32) / 2147483648.0
            
        # 3. Ubah Stereo ke Mono (Kalau mic laptop lu stereo)
        if len(data_suara.shape) > 1:
            data_suara = np.mean(data_suara, axis=1)
            
        # 4. Samakan Frekuensi ke 16000 Hz (Standar mutlak Whisper)
        if sr_asli != 16000:
            data_suara = librosa.resample(data_suara, orig_sr=sr_asli, target_sr=16000)

        # 🔥 BYPASS: Masukkan data langsung ke Whisper tanpa save ke .wav!
        input_features = processor_asr(
            data_suara, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)

        forced_decoder_ids = processor_asr.get_decoder_prompt_ids(language="english", task="transcribe")
        predicted_ids = model_asr.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        teks_transkrip = processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        end_asr = time.time() 
        waktu_asr = end_asr - start_asr
        print(f"   📝 Transkrip : '{teks_transkrip}'")
        print(f"   ⏱️ Waktu ASR : {waktu_asr:.2f} detik")

        # --- TAHAP 2: MT ---
        print("\n🧠 TAHAP 2: Menerjemahkan makna...")
        start_mt = time.time() 
        
        daftar_kalimat = [k.strip() + "." for k in teks_transkrip.split('.') if k.strip()]
        hasil_terjemahan_lengkap = []
        
        for kalimat in daftar_kalimat:
            inputs_mt = tokenizer_mt(kalimat, return_tensors="pt", padding=True).to(device)
            translated_ids = model_mt.generate(**inputs_mt, max_length=512)
            teks_indo_potongan = tokenizer_mt.batch_decode(translated_ids, skip_special_tokens=True)[0].strip()
            hasil_terjemahan_lengkap.append(teks_indo_potongan)
        
        teks_indo = " ".join(hasil_terjemahan_lengkap)
        
        end_mt = time.time() 
        waktu_mt = end_mt - start_mt
        print(f"   🌐 Terjemahan: '{teks_indo}'")
        print(f"   ⏱️ Waktu MT  : {waktu_mt:.2f} detik")

        # --- TAHAP 3: TTS ---
        print("\n🔊 TAHAP 3: Membuat suara balasan...")
        start_tts = time.time() 
        
        file_output = "output_suara_lokal.mp3"
        gTTS(teks_indo, lang="id").save(file_output)
        
        end_tts = time.time() 
        waktu_tts = end_tts - start_tts
        print(f"   ⏱️ Waktu TTS : {waktu_tts:.2f} detik")
        
        waktu_total = waktu_asr + waktu_mt + waktu_tts
        print(f"✅ PROSES SELESAI! (Total Latency AI: {waktu_total:.2f} detik)")
        print("="*40)

        return teks_transkrip, teks_indo, file_output

    except Exception as e:
        print(f"❌ ERROR FATAL DETECTED: {str(e)}")
        return f"Error: {str(e)}", "Sistem gagal memproses", None

# ==========================================
# 4. MEMBANGUN UI WEB GRADIO
# ==========================================
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🎧 Prototipe Smart Headset (Live Web Mode)")
    gr.Markdown("Pencet tombol **Record**, ngomong pakai bahasa Inggris, lalu klik **Terjemahkan**.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎙️ Input Mic")
            # sources=["microphone"] = Browser otomatis pake Mic laptop lu
            # type="filepath" = Gradio otomatis nyimpen wav sementaranya ke temp folder
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Mic Input")
            btn_submit = gr.Button("🚀 Terjemahkan", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🤖 Output Sistem")
            teks_inggris_ui = gr.Textbox(label="1. Transkrip ASR (Inggris)")
            teks_indo_ui = gr.Textbox(label="2. Terjemahan MT (Indonesia)")
            # autoplay=True = Browser otomatis muter suara balasan tanpa perlu klik play
            audio_out = gr.Audio(label="3. Audio Balasan", autoplay=True)

    # Nyambungin tombol ke fungsi AI lu
    btn_submit.click(
        fn=smart_headset_web,
        inputs=audio_in,
        outputs=[teks_inggris_ui, teks_indo_ui, audio_out]
    )

if __name__ == "__main__":
    print("🚀 MENJALANKAN SERVER WEB LOKAL...")
    # Akan memunculkan link http://127.0.0.1:7860/ di terminal lu
    demo.launch(share=False)