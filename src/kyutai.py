#!/usr/bin/env python3

import dataclasses
import itertools
import math
import os
import time
import torch
from typing import List, Tuple
import warnings

import julius
import moshi.models
import sphn
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning)

@dataclasses.dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]

    def __str__(self):
        return f"{self.text} ({self.timestamp[0]:.2f}:{self.timestamp[1]:.2f})"

class MoshiTranscriber:
    def __init__(self, hf_repo="kyutai/stt-1b-en_fr-candle", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hf_repo = hf_repo
        
        print(f"Loading Moshi STT model: {hf_repo}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
            self.hf_repo,
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )
        
        self.mimi = self.info.get_mimi(device=self.device)
        self.tokenizer = self.info.get_text_tokenizer()
        self.lm = self.info.get_moshi(device=self.device, dtype=torch.bfloat16)
        self.lm_gen = moshi.models.LMGen(self.lm, temp=0, temp_text=0.0)
        
        self.audio_silence_prefix_seconds = self.info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.audio_delay_seconds = self.info.stt_config.get("audio_delay_seconds", 5.0)
        self.padding_token_id = self.info.raw_config.get("text_padding_token_id", 3)
        
        print("Moshi STT model loaded successfully!")
    
    def convert_audio_format(self, input_path: str, target_sr: int = None) -> str:
        output_path = "moshi_converted_audio.wav"
        
        print(f"Converting audio: {input_path}")
        
        audio = AudioSegment.from_file(input_path)
        
        print(f"Original: {audio.frame_rate}Hz, {audio.channels} channel(s), {len(audio)/1000:.2f}s")
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            print("Converted to mono")
        
        if target_sr:
            audio = audio.set_frame_rate(target_sr)
            print(f"Resampled to {target_sr}Hz")
        elif audio.frame_rate not in [16000, 22050, 24000, 32000, 44100, 48000]:
            audio = audio.set_frame_rate(24000)
            print("Resampled to 24kHz")
        
        audio.export(output_path, format="wav")
        print(f"Saved as: {output_path}")
        
        return output_path
    
    def tokens_to_timestamped_text(self, text_tokens, frame_rate, 
                                  end_of_padding_id, offset_seconds) -> List[TimestampedText]:
        text_tokens = text_tokens.cpu().view(-1)
        sequence_timestamps = []

        def _tstmp(start_position, end_position):
            return (
                max(0, start_position / frame_rate - offset_seconds),
                max(0, end_position / frame_rate - offset_seconds),
            )

        def _decode(t):
            t = t[t > self.padding_token_id]
            return self.tokenizer.decode(t.numpy().tolist())

        def _decode_segment(start, end):
            nonlocal text_tokens
            nonlocal sequence_timestamps

            text = _decode(text_tokens[start:end])
            words_inside_segment = text.split()

            if len(words_inside_segment) == 0:
                return
            if len(words_inside_segment) == 1:
                sequence_timestamps.append(
                    TimestampedText(text=text, timestamp=_tstmp(start, end))
                )
            else:
                for adjacent_word in words_inside_segment[:-1]:
                    n_tokens = len(self.tokenizer.encode(adjacent_word))
                    sequence_timestamps.append(
                        TimestampedText(
                            text=adjacent_word, timestamp=_tstmp(start, start + n_tokens)
                        )
                    )
                    start += n_tokens

                adjacent_word = words_inside_segment[-1]
                sequence_timestamps.append(
                    TimestampedText(text=adjacent_word, timestamp=_tstmp(start, end))
                )

        (segment_boundaries,) = torch.where(text_tokens == end_of_padding_id)

        if not segment_boundaries.numel():
            return []

        for i in range(len(segment_boundaries) - 1):
            segment_start = int(segment_boundaries[i]) + 1
            segment_end = int(segment_boundaries[i + 1])
            _decode_segment(segment_start, segment_end)

        last_segment_start = segment_boundaries[-1] + 1
        boundary_token = torch.tensor([self.tokenizer.eos_id()])
        (end_of_last_segment,) = torch.where(
            torch.isin(text_tokens[last_segment_start:], boundary_token)
        )

        if not end_of_last_segment.numel():
            last_segment_end = min(text_tokens.shape[-1], last_segment_start + frame_rate)
        else:
            last_segment_end = last_segment_start + end_of_last_segment[0]
        
        _decode_segment(last_segment_start, last_segment_end)

        return sequence_timestamps
    
    def transcribe_audio_chunk(self, audio_chunk: torch.Tensor) -> Tuple[str, List[TimestampedText]]:
        text_tokens_accum = []
        plain_text_parts = []
        
        n_prefix_chunks = math.ceil(self.audio_silence_prefix_seconds * self.mimi.frame_rate)
        n_suffix_chunks = math.ceil(self.audio_delay_seconds * self.mimi.frame_rate)
        silence_chunk = torch.zeros(
            (1, 1, self.mimi.frame_size), dtype=torch.float32, device=self.device
        )
        
        chunks = itertools.chain(
            itertools.repeat(silence_chunk, n_prefix_chunks),
            torch.split(audio_chunk[:, None], self.mimi.frame_size, dim=-1),
            itertools.repeat(silence_chunk, n_suffix_chunks),
        )
        
        with self.mimi.streaming(1), self.lm_gen.streaming(1):
            for audio_segment in chunks:
                audio_tokens = self.mimi.encode(audio_segment)
                text_tokens = self.lm_gen.step(audio_tokens)
                text_token = text_tokens[0, 0, 0].cpu().item()
                
                if text_token not in (0, 3):
                    _text = self.tokenizer.id_to_piece(text_tokens[0, 0, 0].cpu().item())
                    _text = _text.replace("▁", " ")
                    plain_text_parts.append(_text)
                
                text_tokens_accum.append(text_tokens)
        
        plain_text = ''.join(plain_text_parts).strip()
        
        if text_tokens_accum:
            utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
            timestamped_segments = self.tokens_to_timestamped_text(
                utterance_tokens,
                self.mimi.frame_rate,
                end_of_padding_id=0,
                offset_seconds=int(n_prefix_chunks / self.mimi.frame_rate) + self.audio_delay_seconds,
            )
        else:
            timestamped_segments = []
        
        return plain_text, timestamped_segments
    
    def split_long_audio(self, audio_path: str, max_duration: float = 120.0) -> List[str]:
        audio_info = AudioSegment.from_file(audio_path)
        total_duration = len(audio_info) / 1000.0
        
        print(f"Audio duration: {total_duration:.2f} seconds")
        
        if total_duration <= max_duration:
            print("Audio is short enough for single processing")
            return [audio_path]
        
        print(f"Splitting into {max_duration}s chunks...")
        
        chunk_files = []
        chunk_duration_ms = int(max_duration * 1000)
        
        for i in range(0, len(audio_info), chunk_duration_ms):
            end_ms = min(i + chunk_duration_ms, len(audio_info))
            
            if end_ms - i < 5000:
                break
            
            chunk = audio_info[i:end_ms]
            chunk_path = f"moshi_chunk_{len(chunk_files):03d}.wav"
            
            if chunk.channels > 1:
                chunk = chunk.set_channels(1)
            
            chunk.export(chunk_path, format="wav")
            chunk_files.append(chunk_path)
            
            print(f"Created chunk {len(chunk_files)}: {i/1000:.1f}s - {end_ms/1000:.1f}s")
        
        return chunk_files
    
    def transcribe_file(self, input_audio: str, output_file: str = None, 
                       timestamped_output: str = None, max_chunk_duration: float = 90.0,
                       keep_temp_files: bool = False) -> str:
        temp_files = []
        
        try:
            converted_audio = self.convert_audio_format(input_audio)
            temp_files.append(converted_audio)
            
            chunk_files = self.split_long_audio(converted_audio, max_chunk_duration)
            temp_files.extend(chunk_files)
            
            all_transcripts = []
            all_timestamped = []
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"\nProcessing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
                
                try:
                    audio, input_sample_rate = sphn.read(chunk_file)
                    audio = torch.from_numpy(audio).to(self.device)
                    
                    audio = julius.resample_frac(audio, input_sample_rate, self.mimi.sample_rate)
                    
                    if audio.shape[-1] % self.mimi.frame_size != 0:
                        to_pad = self.mimi.frame_size - audio.shape[-1] % self.mimi.frame_size
                        audio = torch.nn.functional.pad(audio, (0, to_pad))
                    
                    plain_text, timestamped_text = self.transcribe_audio_chunk(audio)
                    
                    if plain_text.strip():
                        all_transcripts.append(plain_text)
                        all_timestamped.extend(timestamped_text)
                        print(f"✓ Transcribed: {plain_text[:80]}...")
                    else:
                        print("✗ No transcription generated")
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"✗ Error processing chunk {i+1}: {e}")
                    continue
            
            final_transcript = ' '.join(all_transcripts).strip()
            
            if not final_transcript:
                raise ValueError("No transcription was generated from any chunk")
            
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(final_transcript)
                print(f"Plain transcript saved to: {output_file}")
            
            if timestamped_output and all_timestamped:
                with open(timestamped_output, "w", encoding="utf-8") as f:
                    f.write("Kyutai Moshi STT - Timestamped Transcription\n")
                    f.write(f"Input: {input_audio}\n")
                    f.write("Format: word (start:end)\n\n")
                    for segment in all_timestamped:
                        f.write(str(segment) + "\n")
                print(f"Timestamped transcript saved to: {timestamped_output}")
            
            return final_transcript
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            raise
        
        finally:
            if not keep_temp_files and temp_files:
                print("\nCleaning up temporary files...")
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file) and temp_file != input_audio:
                            os.remove(temp_file)
                            print(f"Removed: {temp_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove {temp_file}: {e}")

def transcribe_streaming_simple(input_audio: str, output_file: str = "moshi_simple_transcript.txt",
                               timestamped_file: str = "moshi_timestamped.txt", 
                               hf_repo: str = "kyutai/stt-1b-en_fr-candle",
                               enable_vad: bool = False) -> str:
    print("=== MOSHI SIMPLE STREAMING TRANSCRIPTION ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not input_audio.lower().endswith('.wav'):
        print("Converting audio to WAV format...")
        converted_path = "temp_converted.wav"
        audio_seg = AudioSegment.from_file(input_audio)
        if audio_seg.channels > 1:
            audio_seg = audio_seg.set_channels(1)
        audio_seg.export(converted_path, format="wav")
        audio_path = converted_path
    else:
        audio_path = input_audio
    
    try:
        info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
            hf_repo if enable_vad else hf_repo,
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )

        mimi = info.get_mimi(device=device)
        tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
        lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

        audio_silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)
        padding_token_id = info.raw_config.get("text_padding_token_id", 3)

        audio, input_sample_rate = sphn.read(audio_path)
        audio = torch.from_numpy(audio).to(device)
        audio = julius.resample_frac(audio, input_sample_rate, mimi.sample_rate)
        
        if audio.shape[-1] % mimi.frame_size != 0:
            to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, to_pad))

        text_tokens_accum = []
        plain_text_parts = []

        n_prefix_chunks = math.ceil(audio_silence_prefix_seconds * mimi.frame_rate)
        n_suffix_chunks = math.ceil(audio_delay_seconds * mimi.frame_rate)
        silence_chunk = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)

        chunks = itertools.chain(
            itertools.repeat(silence_chunk, n_prefix_chunks),
            torch.split(audio[:, None], mimi.frame_size, dim=-1),
            itertools.repeat(silence_chunk, n_suffix_chunks),
        )

        print("Processing audio stream...")
        start_time = time.time()
        nchunks = 0
        
        with mimi.streaming(1), lm_gen.streaming(1):
            for audio_chunk in chunks:
                nchunks += 1
                audio_tokens = mimi.encode(audio_chunk)
                
                if enable_vad:
                    text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)
                    if vad_heads:
                        pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                        if pr_vad > 0.5:
                            plain_text_parts.append(" [end of turn] ")
                else:
                    text_tokens = lm_gen.step(audio_tokens)
                
                text_token = text_tokens[0, 0, 0].cpu().item()
                if text_token not in (0, 3):
                    _text = tokenizer.id_to_piece(text_tokens[0, 0, 0].cpu().item())
                    _text = _text.replace("▁", " ")
                    plain_text_parts.append(_text)
                
                text_tokens_accum.append(text_tokens)

        plain_transcript = ''.join(plain_text_parts).strip()
        
        dt = time.time() - start_time
        print(f"\nProcessed {nchunks} chunks in {dt:.2f} seconds")
        print(f"Processing speed: {nchunks / dt:.2f} steps/second")
        
        if text_tokens_accum:
            utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
            timestamped_segments = []
            
            if hasattr(MoshiTranscriber, 'tokens_to_timestamped_text'):
                transcriber = MoshiTranscriber.__new__(MoshiTranscriber)
                transcriber.tokenizer = tokenizer
                transcriber.padding_token_id = padding_token_id
                timestamped_segments = transcriber.tokens_to_timestamped_text(
                    utterance_tokens, mimi.frame_rate, 0,
                    int(n_prefix_chunks / mimi.frame_rate) + audio_delay_seconds
                )
            
            if timestamped_file and timestamped_segments:
                with open(timestamped_file, "w", encoding="utf-8") as f:
                    f.write("Kyutai Moshi STT - Timestamped Transcription\n")
                    f.write("="*50 + "\n\n")
                    for segment in timestamped_segments:
                        f.write(str(segment) + "\n")
                print(f"Timestamped transcript saved to: {timestamped_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(plain_transcript)
        
        print(f"Plain transcript saved to: {output_file}")
        return plain_transcript
        
    finally:
        if audio_path != input_audio and os.path.exists(audio_path):
            os.remove(audio_path)

def main():
    input_audio = "./audioFile.mp3"
    
    if not os.path.exists(input_audio):
        print(f"Error: Input file {input_audio} not found!")
        print("Please ensure your audio file is named 'audioFile.mp3' in the current directory")
        return
    
    print("=== KYUTAI MOSHI STT TRANSCRIPTION ===")
    
    try:
        print("Attempting chunked transcription...")
        transcriber = MoshiTranscriber()
        
        result = transcriber.transcribe_file(
            input_audio=input_audio,
            output_file="moshi_complete_transcript.txt",
            timestamped_output="moshi_timestamped_transcript.txt",
            max_chunk_duration=90.0,
            keep_temp_files=False
        )
        
        print("\n" + "="*80)
        print("FINAL MOSHI TRANSCRIPTION:")
        print("="*80)
        print(result)
        print("="*80)
        print(f"Word count: {len(result.split())}")
        
    except Exception as e:
        print(f"Chunked transcription failed: {e}")
        print("\nTrying simple streaming transcription...")
        
        try:
            simple_result = transcribe_streaming_simple(
                input_audio, 
                "moshi_simple_transcript.txt",
                "moshi_simple_timestamped.txt"
            )
            
            print("\n" + "="*60)
            print("SIMPLE TRANSCRIPTION RESULT:")
            print("="*60)
            print(simple_result)
            print("="*60)
            print(f"Word count: {len(simple_result.split())}")
            
        except Exception as simple_e:
            print(f"Simple transcription also failed: {simple_e}")

def main_with_vad():
    input_audio = "./audioFile.mp3"
    
    if not os.path.exists(input_audio):
        print(f"Error: {input_audio} not found!")
        return
    
    print("=== MOSHI STT WITH VAD ===")
    
    result = transcribe_streaming_simple(
        input_audio,
        "moshi_vad_transcript.txt", 
        "moshi_vad_timestamped.txt",
        enable_vad=True
    )
    
    print(f"VAD-enabled transcription complete. Word count: {len(result.split())}")

if __name__ == "__main__":
    main()
    
    
    print("\nMoshi STT transcription completed!")
