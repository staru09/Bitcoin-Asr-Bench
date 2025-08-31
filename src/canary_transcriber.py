import torch
from nemo.collections.speechlm2.models import SALM
from pydub import AudioSegment
import os
import warnings
from typing import List
import math

warnings.filterwarnings("ignore", category=UserWarning)

class CanaryTranscriber:
    def __init__(self, model_name="nvidia/canary-qwen-2.5b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Loading model {model_name} on device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        if self.device == "cuda":
            self.model = SALM.from_pretrained(model_name).to(self.device)
            self.model = self.model.half()
            print("Model loaded on GPU with half precision")
        else:
            self.model = SALM.from_pretrained(model_name)
            print("Model loaded on CPU")
        
        print("Model loaded successfully!")
    
    def convert_to_16k_mono(self, input_file: str, output_file: str = "converted.wav") -> str:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Audio file not found: {input_file}")
        
        print(f"Converting audio: {input_file} -> {output_file}")
        
        audio = AudioSegment.from_file(input_file)
        
        original_rate = audio.frame_rate
        original_channels = audio.channels
        original_duration = len(audio) / 1000.0
        
        print(f"Original: {original_rate}Hz, {original_channels} channel(s), {original_duration:.2f}s")
        
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")
        
        print(f"Converted: 16000Hz, 1 channel, {original_duration:.2f}s")
        return output_file
    
    def create_audio_chunks(self, audio_file: str, chunk_duration: float = 30.0, 
                           overlap_duration: float = 3.0) -> List[str]:
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio) / 1000.0 
        
        if total_duration <= chunk_duration:
            print(f"Audio ({total_duration:.1f}s) is short enough for single processing")
            return [audio_file]
        
        print(f"Splitting {total_duration:.1f}s audio into {chunk_duration}s chunks with {overlap_duration}s overlap")
        
        chunk_files = []
        chunk_duration_ms = int(chunk_duration * 1000)
        overlap_ms = int(overlap_duration * 1000)
        step_ms = chunk_duration_ms - overlap_ms
        
        chunk_idx = 0
        for start_ms in range(0, len(audio), step_ms):
            end_ms = min(start_ms + chunk_duration_ms, len(audio))
            
            if end_ms - start_ms < 2000:
                break
            
            chunk = audio[start_ms:end_ms]
            chunk_file = f"chunk_{chunk_idx:03d}.wav"
            chunk.export(chunk_file, format="wav")
            chunk_files.append(chunk_file)
            
            print(f"Created chunk {chunk_idx}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s -> {chunk_file}")
            chunk_idx += 1
        
        return chunk_files
    
    def transcribe_single_file(self, wav_file: str, max_tokens: int = 512) -> str:
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                answer_ids = self.model.generate(
                    prompts=[[
                        {
                            "role": "user",
                            "content": f"Transcribe the following audio clearly and accurately: {self.model.audio_locator_tag}",
                            "audio": [wav_file]
                        }
                    ]],
                    max_new_tokens=max_tokens,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=0.8,
                )
            
            transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            return self.clean_transcript(transcript)
            
        except Exception as e:
            print(f"Error transcribing {wav_file}: {e}")
            return ""
    
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
        
        cleaned_text = ' '.join(cleaned_words)
        
        cleaned_text = cleaned_text.replace(' .', '.')
        cleaned_text = cleaned_text.replace(' ,', ',')
        cleaned_text = cleaned_text.replace(' ?', '?')
        cleaned_text = cleaned_text.replace(' !', '!')
        
        return cleaned_text.strip()
    
    def is_repetitive_transcript(self, text: str, threshold: float = 0.5) -> bool:
        if len(text) < 20:
            return False
        
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return diversity_ratio < threshold
    
    def merge_chunk_transcripts(self, transcripts: List[str]) -> str:
        if not transcripts:
            return ""
        
        valid_transcripts = [
            t for t in transcripts 
            if t and not self.is_repetitive_transcript(t)
        ]
        
        if not valid_transcripts:
            return ""
        
        if len(valid_transcripts) == 1:
            return valid_transcripts[0]
        
        merged_parts = [valid_transcripts[0]]
        
        for i in range(1, len(valid_transcripts)):
            current = valid_transcripts[i]
            previous = merged_parts[-1]
            
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
    
    def cleanup_temp_files(self, chunk_files: List[str]):
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    print(f"Removed temporary file: {chunk_file}")
            except Exception as e:
                print(f"Warning: Could not remove {chunk_file}: {e}")
    
    def transcribe_file(self, audio_path: str, chunk_duration: float = 25.0, 
                       overlap_duration: float = 3.0, output_file: str = None,
                       keep_chunks: bool = False) -> str:
        chunk_files = []
        
        try:
            converted_file = self.convert_to_16k_mono(audio_path)
            
            chunk_files = self.create_audio_chunks(
                converted_file, chunk_duration, overlap_duration
            )
            
            print(f"\nTranscribing {len(chunk_files)} chunk(s)...")
            transcripts = []
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"\nProcessing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
                
                transcript = self.transcribe_single_file(chunk_file)
                
                if transcript and not self.is_repetitive_transcript(transcript):
                    transcripts.append(transcript)
                    print(f"✓ Success: {transcript[:100]}...")
                else:
                    print(f"✗ Skipped repetitive/empty transcript")
            
            if not transcripts:
                raise ValueError("No valid transcripts generated from any chunks")
            
            print(f"\nMerging {len(transcripts)} valid transcript(s)...")
            final_transcript = self.merge_chunk_transcripts(transcripts)
            
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_transcript)
                print(f"Final transcript saved to: {output_file}")
            
            return final_transcript
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            raise
        
        finally:
            if not keep_chunks and chunk_files:
                print("\nCleaning up temporary files...")
                self.cleanup_temp_files(chunk_files)
                
                if "converted.wav" in [converted_file] and converted_file != audio_path:
                    try:
                        os.remove(converted_file)
                        print(f"Removed temporary file: {converted_file}")
                    except:
                        pass

def main():
    input_audio = "./audioFile.wav"
    output_file = "canary_complete_transcription.txt"
    
    print("Initializing NVIDIA Canary transcriber...")
    transcriber = CanaryTranscriber()
    
    print("\nStarting transcription process...")
    try:
        result = transcriber.transcribe_file(
            audio_path=input_audio,
            chunk_duration=25.0,
            overlap_duration=3.0,
            output_file=output_file,
            keep_chunks=False
        )
        
        print("\n" + "="*80)
        print("FINAL TRANSCRIPTION RESULT:")
        print("="*80)
        print(result)
        print("="*80)
        
        backup_file = f"canary_backup_{hash(result) % 10000}.txt"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write("NVIDIA Canary Transcription Result\n")
            f.write(f"Input file: {input_audio}\n")
            f.write(f"Model: nvidia/canary-qwen-2.5b\n")
            f.write(f"Processing method: Chunked\n\n")
            f.write("TRANSCRIPT:\n")
            f.write(result)
        
        print(f"Backup saved to: {backup_file}")
        print(f"Word count: {len(result.split())}")
        
        return result
        
    except Exception as e:
        print(f"Transcription process failed: {e}")
        return None

def transcribe_simple(input_audio: str, output_file: str = "simple_transcript.txt") -> str:
    def convert_to_16k_mono(input_file, output_file="converted_simple.wav"):
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")
        return output_file
    
    wav_file = convert_to_16k_mono(input_audio)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    
    if device == "cuda":
        model = model.to(device).half()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Model loaded on CPU")
    
    with torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad():
        answer_ids = model.generate(
            prompts=[[
                {
                    "role": "user",
                    "content": f"Transcribe the following audio completely and accurately: {model.audio_locator_tag}",
                    "audio": [wav_file]
                }
            ]],
            max_new_tokens=512,
            min_new_tokens=10,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    transcript = model.tokenizer.ids_to_text(answer_ids[0].cpu())
    
    transcript = ' '.join(transcript.split())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    try:
        os.remove(wav_file)
    except:
        pass
    
    print(f"Simple transcript saved to {output_file}")
    return transcript

if __name__ == "__main__":
    print("=== CHUNKED TRANSCRIPTION (Recommended for long audio) ===")
    main()
    
    print("\n" + "="*50 + "\n")
    
