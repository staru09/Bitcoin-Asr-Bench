import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import os
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

class GraniteTranscriber:
    def __init__(self, model_name="ibm-granite/granite-speech-3.3-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Loading model {model_name} on device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, 
            device_map=self.device, 
            torch_dtype=torch.bfloat16
        )
        
        print("Model loaded successfully!")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Loading audio from: {audio_path}")
        wav, sr = torchaudio.load(audio_path, normalize=True)
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            print("Converted stereo to mono")
        
        duration = wav.shape[1] / sr
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        return wav, sr
    
    def transcribe_chunk(self, chunk: torch.Tensor, chunk_idx: int) -> str:
        system_prompt = ("You are Granite, an AI assistant developed by IBM. "
                        "Transcribe the following audio accurately into readable text.")
        user_prompt = "<|audio|>Please transcribe this audio clearly and completely."
        
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.processor(
            prompt, chunk, device=self.device, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            model_outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=200,
                min_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                length_penalty=0.8,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:]
        chunk_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return self.clean_text(chunk_text)
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(cleaned_words) < 2 or not all(
                w.lower() == word.lower() for w in cleaned_words[-2:]
            ):
                cleaned_words.append(word)
        
        cleaned_text = ' '.join(cleaned_words)
        
        cleaned_text = cleaned_text.replace('  ', ' ')
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def is_repetitive(self, text: str, threshold: float = 0.6) -> bool:
        if len(text) < 20:
            return False
        
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        if diversity_ratio < threshold:
            return True
        
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        max_word_freq = max(word_count.values()) / len(words)
        if max_word_freq > 0.3:
            return True
        
        return False
    
    def merge_overlapping_transcriptions(self, transcriptions: List[str]) -> str:
        if not transcriptions:
            return ""
        
        if len(transcriptions) == 1:
            return transcriptions[0]
        
        merged = []
        
        for i, transcription in enumerate(transcriptions):
            if not transcription:
                continue
            
            transcription = transcription.strip()
            
            if i == 0:
                merged.append(transcription)
            else:
                prev_words = merged[-1].split()[-10:]
                curr_words = transcription.split()
                
                best_overlap = 0
                for j in range(1, min(len(prev_words), len(curr_words)) + 1):
                    if prev_words[-j:] == curr_words[:j]:
                        best_overlap = j
                
                if best_overlap > 0:
                    merged.append(' '.join(curr_words[best_overlap:]))
                else:
                    merged.append(transcription)
        
        return ' '.join(merged)
    
    def transcribe_file(self, audio_path: str, chunk_duration: int = 30, 
                       overlap_duration: int = 3, output_file: str = None) -> str:
        try:
            wav, sr = self.preprocess_audio(audio_path)
            total_duration = wav.shape[1] / sr
            
            if total_duration <= chunk_duration:
                print("Audio is short enough for single-pass transcription")
                result = self.transcribe_chunk(wav, 0)
            else:
                print(f"Processing {total_duration:.1f}s audio in {chunk_duration}s chunks")
                result = self.transcribe_long_audio(wav, sr, chunk_duration, overlap_duration)
            
            result = self.clean_text(result)
            
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"Transcription saved to: {output_file}")
            
            return result
            
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            raise
    
    def transcribe_long_audio(self, wav: torch.Tensor, sr: int, 
                             chunk_duration: int, overlap_duration: int) -> str:
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        
        total_samples = wav.shape[1]
        transcriptions = []
        
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = wav[:, start:end]
            
            if chunk.shape[1] < sr * 2:
                break
            
            chunk_idx = len(transcriptions) + 1
            start_time = start / sr
            end_time = end / sr
            
            print(f"Processing chunk {chunk_idx}: {start_time:.1f}s - {end_time:.1f}s")
            
            try:
                chunk_text = self.transcribe_chunk(chunk, chunk_idx)
                
                if chunk_text and not self.is_repetitive(chunk_text):
                    transcriptions.append(chunk_text)
                    print(f"✓ Chunk {chunk_idx}: {chunk_text[:80]}...")
                else:
                    print(f"✗ Skipped repetitive/empty chunk {chunk_idx}")
                    
            except Exception as e:
                print(f"✗ Error processing chunk {chunk_idx}: {e}")
                continue
        
        if not transcriptions:
            raise ValueError("No successful transcriptions generated")
        
        return self.merge_overlapping_transcriptions(transcriptions)

def main():
    audio_path = "./audioFile.wav"
    output_file = "granite_complete_transcription.txt"
    
    transcriber = GraniteTranscriber()
    
    print("Starting transcription...")
    try:
        result = transcriber.transcribe_file(
            audio_path=audio_path,
            chunk_duration=25,
            overlap_duration=3,
            output_file=output_file
        )
        
        print("\n" + "="*60)
        print("FINAL TRANSCRIPTION:")
        print("="*60)
        print(result)
        print("="*60)
        
        timestamp_file = f"transcription_{torch.randint(1000, 9999, (1,)).item()}.txt"
        with open(timestamp_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription completed successfully\n")
            f.write(f"Audio file: {audio_path}\n")
            f.write(f"Model: {transcriber.model_name}\n\n")
            f.write("TRANSCRIPTION:\n")
            f.write(result)
        
        print(f"Backup saved to: {timestamp_file}")
        
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None

if __name__ == "__main__":
    main()
