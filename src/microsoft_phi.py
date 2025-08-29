#!/usr/bin/env python3

import torch
import os
import io
import soundfile as sf
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from pydub import AudioSegment
import warnings
from typing import List, Tuple
import math

warnings.filterwarnings("ignore", category=UserWarning)

class Phi4Transcriber:
    def __init__(self, model_path="microsoft/Phi-4-multimodal-instruct"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Phi-4 multimodal model: {model_path}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'eager',
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU with flash attention")
        else:
            print("Model loaded on CPU")
        
                self.generation_config = GenerationConfig.from_pretrained(model_path)
        
                self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
        
        print("Phi-4 model initialized successfully!")
    
    def convert_audio_format(self, input_path: str, target_sr: int = 16000) -> str:
        output_path = "phi4_converted_audio.wav"
        
        print(f"Converting audio: {input_path}")
        
                audio = AudioSegment.from_file(input_path)
        
        print(f"Original: {audio.frame_rate}Hz, {audio.channels} channel(s), {len(audio)/1000:.2f}s")
        
                if audio.channels > 1:
            audio = audio.set_channels(1)
            print("Converted to mono")
        
                audio = audio.set_frame_rate(target_sr)
        print(f"Resampled to {target_sr}Hz")
        
                audio.export(output_path, format="wav")
        print(f"Saved as: {output_path}")
        
        return output_path
    
    def split_audio_for_processing(self, audio_path: str, max_duration: float = 30.0, 
                                  overlap_duration: float = 2.0) -> List[str]:
                with sf.SoundFile(audio_path) as f:
            total_duration = len(f) / f.samplerate
        
        print(f"Audio duration: {total_duration:.2f} seconds")
        
        if total_duration <= max_duration:
            print("Audio is short enough for single processing")
            return [audio_path]
        
        print(f"Splitting into {max_duration}s chunks with {overlap_duration}s overlap...")
        
                audio = AudioSegment.from_file(audio_path)
        
        chunk_ms = int(max_duration * 1000)
        overlap_ms = int(overlap_duration * 1000)
        step_ms = chunk_ms - overlap_ms
        
        chunk_files = []
        chunk_idx = 0
        
        for start_ms in range(0, len(audio), step_ms):
            end_ms = min(start_ms + chunk_ms, len(audio))
            
            # Skip very short chunks
            if end_ms - start_ms < 3000:
                break
            
            chunk = audio[start_ms:end_ms]
            chunk_path = f"phi4_chunk_{chunk_idx:03d}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_files.append(chunk_path)
            
            print(f"Created chunk {chunk_idx}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            chunk_idx += 1
        
        return chunk_files
    
    def transcribe_single_chunk(self, audio_path: str) -> str:
        try:
            # Load audio
            audio, samplerate = sf.read(audio_path)
            
            # Create prompt
            speech_prompt = "Transcribe the audio to text. Provide only the spoken text without any additional formatting or separators."
            prompt = f'{self.user_prompt}<|audio_1|>{speech_prompt}{self.prompt_suffix}{self.assistant_prompt}'
            
                        inputs = self.processor(
                text=prompt, 
                audios=[(audio, samplerate)], 
                return_tensors='pt'
            ).to(self.device)
            
                        if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
                        with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    generation_config=self.generation_config,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
                        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return self.clean_transcript(response)
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""
    
    def clean_transcript(self, text: str) -> str:
        if not text:
            return ""
        
                text = text.strip()
        
                text = text.replace('<sep>', '').replace('<SEP>', '')
        
                text = ' '.join(text.split())
        
                words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(cleaned_words) < 2 or not all(
                w.lower() == word.lower() for w in cleaned_words[-2:]
            ):
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def is_repetitive(self, text: str, threshold: float = 0.5) -> bool:
        if len(text) < 20:
            return False
        
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return diversity_ratio < threshold
    
    def merge_chunk_transcripts(self, transcripts: List[str]) -> str:
                valid_transcripts = [
            self.clean_transcript(t) for t in transcripts 
            if t.strip() and not self.is_repetitive(t)
        ]
        
        if not valid_transcripts:
            return ""
        
        if len(valid_transcripts) == 1:
            return valid_transcripts[0]
        
                merged_parts = [valid_transcripts[0]]
        
        for i in range(1, len(valid_transcripts)):
            current = valid_transcripts[i]
            previous = merged_parts[-1]
            
            # Look for overlap in last few words
            prev_words = previous.split()[-8:]
            curr_words = current.split()
            
            best_overlap = 0
            for j in range(1, min(len(prev_words), len(curr_words)) + 1):
                if prev_words[-j:] == curr_words[:j]:
                    best_overlap = j
            
                        if best_overlap > 0:
                non_overlap = ' '.join(curr_words[best_overlap:])
                if non_overlap.strip():
                    merged_parts.append(non_overlap)
            else:
                merged_parts.append(current)
        
        return ' '.join(merged_parts)
    
    def cleanup_temp_files(self, file_list: List[str]):
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    def transcribe_file(self, input_audio: str, max_chunk_duration: float = 25.0,
                       overlap_duration: float = 2.0, output_file: str = None,
                       keep_temp_files: bool = False) -> str:
        temp_files = []
        
        try:
            # Step 1: Convert audio to required format
            print("Step 1: Converting audio format...")
            converted_audio = self.convert_audio_format(input_audio)
            temp_files.append(converted_audio)
            
            # Step 2: Split into chunks if needed
            print("\nStep 2: Splitting audio...")
            chunk_files = self.split_audio_for_processing(
                converted_audio, max_chunk_duration, overlap_duration
            )
            temp_files.extend(chunk_files)
            
            # Step 3: Transcribe each chunk
            print(f"\nStep 3: Transcribing {len(chunk_files)} chunk(s)...")
            transcripts = []
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
                
                transcript = self.transcribe_single_chunk(chunk_file)
                
                if transcript and not self.is_repetitive(transcript):
                    transcripts.append(transcript)
                    print(f"✓ Success: {transcript[:80]}...")
                else:
                    print(f"✗ Skipped repetitive/empty result")
            
            # Step 4: Merge results
            print(f"\nStep 4: Merging {len(transcripts)} valid transcript(s)...")
            final_transcript = self.merge_chunk_transcripts(transcripts)
            
            if not final_transcript:
                raise ValueError("No valid transcripts were generated")
            
            # Step 5: Save results
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_transcript)
                print(f"Transcript saved to: {output_file}")
                
                # Save debug info with individual chunks
                debug_file = output_file.replace('.txt', '_debug.txt')
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write("Phi-4 Multimodal ASR Debug Information\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Input file: {input_audio}\n")
                    f.write(f"Chunks processed: {len(chunk_files)}\n")
                    f.write(f"Valid transcripts: {len(transcripts)}\n\n")
                    
                    for i, transcript in enumerate(transcripts):
                        f.write(f"=== CHUNK {i+1} ===\n")
                        f.write(transcript + "\n\n")
                    
                    f.write("=== MERGED RESULT ===\n")
                    f.write(final_transcript)
                
                print(f"Debug info saved to: {debug_file}")
            
            return final_transcript
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            raise
        
        finally:
            # Cleanup temporary files unless requested to keep them
            if not keep_temp_files and temp_files:
                print("\nCleaning up temporary files...")
                self.cleanup_temp_files(temp_files)

def transcribe_simple_phi4(input_audio: str, output_file: str = "phi4_simple_transcript.txt") -> str:
    print("=== PHI-4 SIMPLE TRANSCRIPTION ===")
    
        model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if torch.cuda.is_available() else 'eager',
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    generation_config = GenerationConfig.from_pretrained(model_path)
    
        if not input_audio.endswith('.wav'):
        print("Converting audio format...")
        audio_segment = AudioSegment.from_file(input_audio)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(16000)
        converted_path = "temp_converted.wav"
        audio_segment.export(converted_path, format="wav")
        audio_path = converted_path
    else:
        audio_path = input_audio
    
    try:
        # Load audio
        audio, samplerate = sf.read(audio_path)
        
        max_samples = int(60 * samplerate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print("Audio truncated to 60 seconds for simple mode")
        
        # Create prompt
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        speech_prompt = "Transcribe the audio to text clearly and accurately. Provide only the spoken words."
        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
        
        print("Processing audio...")
        
                inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
                with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=800,
                min_new_tokens=5,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                generation_config=generation_config,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
            )
        
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Clean response
        response = response.strip().replace('<sep>', '').replace('<SEP>', '')
        response = ' '.join(response.split())
        
                with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
        
        print(f"Simple transcription saved to: {output_file}")
        print(f"Result preview: {response[:200]}...")
        
        return response
        
    finally:
        if audio_path != input_audio and os.path.exists(audio_path):
            os.remove(audio_path)

def main():
    # Configuration
    input_audio = "./audioFile.wav"
    output_file = "phi4_complete_transcription.txt"
    
    print("=== PHI-4 MULTIMODAL ASR TRANSCRIPTION ===")
    
        if not os.path.exists(input_audio):
        print(f"Error: Input file {input_audio} not found!")
        print("Please ensure your audio file is named 'audioFile.wav' and in the current directory")
        return
    
        transcriber = Phi4Transcriber()
    
        try:
        result = transcriber.transcribe_file(
            input_audio=input_audio,
            max_chunk_duration=25.0,
            overlap_duration=2.0,
            output_file=output_file,
            keep_temp_files=False
        )
        
        print("\n" + "="*80)
        print("FINAL PHI-4 ASR TRANSCRIPTION:")
        print("="*80)
        print(result)
        print("="*80)
        print(f"Total words: {len(result.split())}")
        print(f"Character count: {len(result)}")
        
                backup_file = f"phi4_backup_{abs(hash(result)) % 10000}.txt"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write("Microsoft Phi-4 Multimodal ASR Transcription\n")
            f.write(f"Model: microsoft/Phi-4-multimodal-instruct\n")
            f.write(f"Input: {input_audio}\n")
            f.write(f"Device: {transcriber.device}\n")
            f.write(f"Processing: Chunked ({25.0}s chunks, {2.0}s overlap)\n\n")
            f.write("TRANSCRIPT:\n")
            f.write(result)
        
        print(f"Backup with metadata saved to: {backup_file}")
        
    except Exception as e:
        print(f"Transcription failed: {e}")
        print("\nTrying simple mode for shorter audio...")
        
                try:
            simple_result = transcribe_simple_phi4(input_audio, "phi4_simple_fallback.txt")
            print("Simple transcription completed successfully!")
        except Exception as simple_e:
            print(f"Simple transcription also failed: {simple_e}")

def main_simple():
    input_audio = "./audioFile.wav"
    
    if not os.path.exists(input_audio):
        print(f"Error: Input file {input_audio} not found!")
        return
    
    result = transcribe_simple_phi4(input_audio, "phi4_simple_transcript.txt")
    print(f"Simple transcription complete. Word count: {len(result.split())}")

if __name__ == "__main__":
    main()
    
    
    print("\nTranscription process completed!")
