#!/usr/bin/env python3

import os
import torch
from pydub import AudioSegment
import nemo.collections.asr as nemo_asr
import warnings
from typing import List
import tempfile

warnings.filterwarnings("ignore", category=UserWarning)

class ParakeetTranscriber:
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Initializing Parakeet model: {model_name}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.asr_model = self.asr_model.to(self.device)
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
        
        self.asr_model.eval()
        print("Model initialized successfully!")
    
    def convert_to_mono_16k(self, input_file: str, output_file: str = None) -> str:
        if output_file is None:
            output_file = "converted_mono.wav"
        
        print(f"Converting {input_file} to mono 16kHz format...")
        
        audio = AudioSegment.from_file(input_file)
        
        print(f"Original: {audio.frame_rate}Hz, {audio.channels} channel(s), {len(audio)/1000:.2f}s")
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        audio = audio.set_frame_rate(16000)
        
        audio.export(output_file, format="wav")
        
        print(f"Converted to: 16000Hz, 1 channel, saved as {output_file}")
        return output_file
    
    def split_audio_chunks(self, input_file: str, chunk_duration: float = 60.0, 
                          overlap_duration: float = 2.0) -> List[str]:
        audio = AudioSegment.from_file(input_file)
        total_duration = len(audio) / 1000.0
        
        print(f"Audio duration: {total_duration:.2f} seconds")
        
        if total_duration <= chunk_duration:
            print("Audio is short enough for single processing")
            return [input_file]
        
        print(f"Splitting into {chunk_duration}s chunks with {overlap_duration}s overlap...")
        
        chunk_ms = int(chunk_duration * 1000)
        overlap_ms = int(overlap_duration * 1000)
        step_ms = chunk_ms - overlap_ms
        
        chunk_paths = []
        chunk_idx = 0
        
        for start_ms in range(0, len(audio), step_ms):
            end_ms = min(start_ms + chunk_ms, len(audio))
            
            if end_ms - start_ms < 3000:
                break
            
            chunk = audio[start_ms:end_ms]
            chunk_path = f"chunk_{chunk_idx:03d}.wav"
            
            if chunk.channels > 1:
                chunk = chunk.set_channels(1)
            chunk = chunk.set_frame_rate(16000)
            
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
            
            print(f"Created chunk {chunk_idx}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
            chunk_idx += 1
        
        print(f"Created {len(chunk_paths)} chunks")
        return chunk_paths
    
    def transcribe_batch(self, audio_files: List[str], batch_size: int = 1) -> List[str]:
        transcriptions = []
        
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{math.ceil(len(audio_files)/batch_size)}: {batch_files}")
            
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                results = self.asr_model.transcribe(
                    batch_files,
                    batch_size=batch_size,
                    return_hypotheses=True
                )
                
                for j, result in enumerate(results):
                    if result and hasattr(result, 'text') and result.text:
                        text = result.text.strip()
                        transcriptions.append(text)
                        print(f"✓ Transcribed chunk {i+j}: {text[:80]}...")
                    else:
                        transcriptions.append("")
                        print(f"✗ Empty result for chunk {i+j}")
                
            except Exception as e:
                print(f"✗ Error processing batch {batch_files}: {e}")
                transcriptions.extend([""] * len(batch_files))
        
        return transcriptions
    
    def clean_transcript(self, text: str) -> str:
        if not text:
            return ""
        
        text = ' '.join(text.split())
        
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(cleaned_words) < 2 or not all(
                w.lower() == word.lower() for w in cleaned_words[-2:]
            ):
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def merge_transcripts(self, transcripts: List[str], overlap_duration: float = 2.0) -> str:
        valid_transcripts = [self.clean_transcript(t) for t in transcripts if t.strip()]
        
        if not valid_transcripts:
            return ""
        
        if len(valid_transcripts) == 1:
            return valid_transcripts[0]
        
        merged_parts = [valid_transcripts[0]]
        
        for i in range(1, len(valid_transcripts)):
            current = valid_transcripts[i]
            
            prev_words = merged_parts[-1].split()[-5:]
            curr_words = current.split()
            
            overlap_found = False
            for j in range(1, min(len(prev_words), len(curr_words)) + 1):
                if prev_words[-j:] == curr_words[:j]:
                    non_overlap = ' '.join(curr_words[j:])
                    if non_overlap.strip():
                        merged_parts.append(non_overlap)
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged_parts.append(current)
        
        return ' '.join(merged_parts)
    
    def cleanup_chunks(self, chunk_files: List[str]):
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    print(f"Removed: {chunk_file}")
            except Exception as e:
                print(f"Warning: Could not remove {chunk_file}: {e}")
    
    def transcribe_file(self, audio_path: str, chunk_duration: float = 45.0,
                       overlap_duration: float = 3.0, batch_size: int = 1,
                       output_file: str = None, keep_chunks: bool = False) -> str:
        chunk_files = []
        converted_file = None
        
        try:
            converted_file = self.convert_to_mono_16k(audio_path)
            
            chunk_files = self.split_audio_chunks(
                converted_file, chunk_duration, overlap_duration
            )
            
            print(f"\nTranscribing {len(chunk_files)} chunk(s) with batch size {batch_size}...")
            transcripts = self.transcribe_batch(chunk_files, batch_size)
            
            print("\nMerging transcripts...")
            final_transcript = self.merge_transcripts(transcripts, overlap_duration)
            
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_transcript)
                print(f"Final transcript saved to: {output_file}")
            
            chunks_file = output_file.replace('.txt', '_chunks.txt') if output_file else 'chunks_debug.txt'
            with open(chunks_file, "w", encoding="utf-8") as f:
                for i, transcript in enumerate(transcripts):
                    f.write(f"=== CHUNK {i+1} ===\n")
                    f.write(transcript + "\n\n")
            
            print(f"Individual chunks saved to: {chunks_file}")
            return final_transcript
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            raise
        
        finally:
            if not keep_chunks:
                if chunk_files:
                    print("\nCleaning up temporary chunk files...")
                    self.cleanup_chunks(chunk_files)
                
                if converted_file and converted_file != audio_path and os.path.exists(converted_file):
                    try:
                        os.remove(converted_file)
                        print(f"Removed temporary file: {converted_file}")
                    except:
                        pass

# Standalone functions for backward compatibility
def split_audio(input_file, chunk_ms=60000):
    audio = AudioSegment.from_file(input_file)
    
        if audio.channels > 1:
        audio = audio.set_channels(1)
    
        audio = audio.set_frame_rate(16000)
    
    chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
    
    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        path = f"chunk_{idx}.wav"
        chunk.export(path, format="wav")
        chunk_paths.append(path)
    
    return chunk_paths

def transcribe_and_save(audio_files, output_file, batch_size=1, timestamps=True):
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    
    if torch.cuda.is_available():
        asr_model = asr_model.to("cuda")
        print("Model loaded on GPU")
    
    all_transcriptions = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
        
        try:
                        if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results = asr_model.transcribe(
                [audio_file],
                batch_size=batch_size, 
                return_hypotheses=True
            )
            
            if results and len(results) > 0 and hasattr(results[0], 'text'):
                text = results[0].text.strip()
                all_transcriptions.append(text)
                print(f"✓ Transcribed: {text[:80]}...")
            else:
                all_transcriptions.append("")
                print("✗ Empty result")
                
        except Exception as e:
            print(f"✗ Error transcribing {audio_file}: {e}")
            all_transcriptions.append("")
    
        with open(output_file, "w", encoding="utf-8") as f:
        for line in all_transcriptions:
            f.write(line + "\n")
    
    return all_transcriptions

def main():
    # Configuration
    input_audio = "./audioFile.wav"
    output_file = "parakeet_complete_transcription.txt"
    
    # Initialize transcriber
    print("Initializing NVIDIA Parakeet transcriber...")
    transcriber = ParakeetTranscriber()
    try:
        result = transcriber.transcribe_file(
            audio_path=input_audio,
            chunk_duration=45.0,
            overlap_duration=3.0,
            batch_size=1,
            output_file=output_file,
            keep_chunks=False
        )
        
        print("\n" + "="*80)
        print("FINAL TRANSCRIPTION:")
        print("="*80)
        print(result)
        print("="*80)
        print(f"Word count: {len(result.split())}")
        
                backup_file = f"parakeet_backup_{hash(result) % 10000}.txt"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write("NVIDIA Parakeet ASR Transcription Result\n")
            f.write(f"Model: nvidia/parakeet-tdt-0.6b-v2\n")
            f.write(f"Input: {input_audio}\n")
            f.write(f"Device: {transcriber.device}\n\n")
            f.write("TRANSCRIPT:\n")
            f.write(result)
        
        print(f"Backup saved to: {backup_file}")
        
    except Exception as e:
        print(f"Transcription failed: {e}")

def main_legacy():
    print("=== LEGACY APPROACH (Fixed) ===")
    
    input_audio = "./audioFile.wav"  # Original input file
    output_file = "transcriptions_parakeet_legacy.txt"
    
    try:
        print("Preprocessing audioFile.wav...")
        audio = AudioSegment.from_file(input_audio)
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            print("Converted to mono")
        
        audio = audio.set_frame_rate(16000)
        print("Converted to 16kHz")
        
        preprocessed_file = "preprocessed_audio.wav"
        audio.export(preprocessed_file, format="wav")
        print(f"Preprocessed audio saved as: {preprocessed_file}")
        
        print("Splitting audio into chunks...")
        chunks = split_audio(preprocessed_file, chunk_ms=45000)  # 45-second chunks
        
        print(f"Created {len(chunks)} chunks. Starting transcription...")
        transcriptions = transcribe_and_save(chunks, output_file, batch_size=1)
        
        merged_transcript = ' '.join([t for t in transcriptions if t.strip()])
        
        merged_file = "merged_" + output_file
        with open(merged_file, "w", encoding="utf-8") as f:
            f.write(merged_transcript)
        
        print(f"Individual transcriptions saved to: {output_file}")
        print(f"Merged transcript saved to: {merged_file}")
        print("Preview:", merged_transcript[:300], "...")
        
        for chunk in chunks:
            try:
                os.remove(chunk)
            except:
                pass
        
        try:
            os.remove(preprocessed_file)
            print(f"Removed preprocessed file: {preprocessed_file}")
        except:
            pass
        
        return merged_transcript
        
    except Exception as e:
        print(f"Legacy transcription failed: {e}")

# Additional ASR utility functions for specialized use cases
def transcribe_with_timestamps(input_audio: str, output_file: str = "timestamped_transcript.txt"):
    print("=== TRANSCRIPTION WITH TIMESTAMPS ===")
    
    transcriber = ParakeetTranscriber()
    
        converted_file = transcriber.convert_to_mono_16k(input_audio, "temp_timestamped.wav")
    
        chunk_files = transcriber.split_audio_chunks(converted_file, chunk_duration=15.0, overlap_duration=1.0)
    
    timestamped_results = []
    
    try:
        for i, chunk_file in enumerate(chunk_files):
            start_time = i * 14.0
            
            print(f"Processing chunk {i+1}/{len(chunk_files)} (t={start_time:.1f}s)")
            
            # Transcribe chunk
            results = transcriber.asr_model.transcribe(
                [chunk_file],
                batch_size=1,
                return_hypotheses=True
            )
            
            if results and len(results) > 0 and hasattr(results[0], 'text'):
                text = results[0].text.strip()
                if text:
                    timestamped_results.append(f"[{start_time:.1f}s] {text}")
                    print(f"✓ {start_time:.1f}s: {text[:60]}...")
            else:
                print(f"✗ No result for chunk at {start_time:.1f}s")
        
                with open(output_file, "w", encoding="utf-8") as f:
            f.write("NVIDIA Parakeet ASR - Timestamped Transcription\n")
            f.write(f"Input: {input_audio}\n")
            f.write("Format: [timestamp] transcript\n\n")
            for result in timestamped_results:
                f.write(result + "\n")
        
        print(f"Timestamped transcript saved to: {output_file}")
        
                clean_transcript = ' '.join([line.split('] ', 1)[1] for line in timestamped_results if '] ' in line])
        clean_file = output_file.replace('.txt', '_clean.txt')
        with open(clean_file, "w", encoding="utf-8") as f:
            f.write(clean_transcript)
        
        print(f"Clean transcript saved to: {clean_file}")
        
        return timestamped_results
        
    finally:
                transcriber.cleanup_chunks(chunk_files)
        if os.path.exists(converted_file):
            os.remove(converted_file)

def batch_transcribe_directory(input_dir: str, output_dir: str = "batch_transcripts"):
    print("=== BATCH TRANSCRIPTION ===")
    
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
    
        audio_files = []
    for ext in audio_extensions:
        audio_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize transcriber once
    transcriber = ParakeetTranscriber()
    results_summary = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n--- Processing {i+1}/{len(audio_files)}: {audio_file} ---")
        
        input_path = os.path.join(input_dir, audio_file)
        output_name = os.path.splitext(audio_file)[0] + "_transcript.txt"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            result = transcriber.transcribe_file(
                audio_path=input_path,
                chunk_duration=30.0,
                overlap_duration=2.0,
                output_file=output_path,
                keep_chunks=False
            )
            
            results_summary.append({
                'file': audio_file,
                'status': 'success',
                'output': output_path,
                'word_count': len(result.split()),
                'preview': result[:100] + '...' if len(result) > 100 else result
            })
            
            print(f"✓ Successfully processed: {audio_file}")
            
        except Exception as e:
            print(f"✗ Failed to process {audio_file}: {e}")
            results_summary.append({
                'file': audio_file,
                'status': 'failed',
                'error': str(e)
            })
    
        summary_file = os.path.join(output_dir, "batch_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("NVIDIA Parakeet ASR - Batch Transcription Summary\n")
        f.write("="*60 + "\n\n")
        
        successful = sum(1 for r in results_summary if r['status'] == 'success')
        failed = len(results_summary) - successful
        
        f.write(f"Total files: {len(results_summary)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        for result in results_summary:
            f.write(f"File: {result['file']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"Output: {result['output']}\n")
                f.write(f"Words: {result['word_count']}\n")
                f.write(f"Preview: {result['preview']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\nBatch processing complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"Successful: {successful}/{len(audio_files)}")

if __name__ == "__main__":
    import math
    
        
    print("=== NVIDIA PARAKEET ASR TRANSCRIPTION ===")
    print("Available modes:")
    print("1. Standard transcription (recommended)")
    print("2. Legacy mode")
    print("3. Timestamped transcription") 
    print("4. Batch directory transcription")
    
        print("\n>>> Running standard transcription mode <<<")
    main()
    
