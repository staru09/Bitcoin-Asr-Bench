# ASR Benchmark Suite

A comprehensive benchmarking framework for evaluating Automatic Speech Recognition (ASR) models across multiple state-of-the-art architectures.

## Overview

This repository provides scripts and tools to transcribe audio files using various ASR models and evaluate their performance using Word Error Rate (WER) and other metrics. The benchmark currently supports four major ASR model families with GPU acceleration and chunked processing for long audio files.

## Supported Models

| Model | Provider | Architecture | WER (Sample) | Notes |
|-------|----------|-------------|--------------|-------|
| **Moshi STT** | Kyutai | Streaming STT | 4.31% | Best overall performance |
| **Parakeet** | NVIDIA | RNN-T | 9.06% | Good balance of speed/accuracy |
| **Canary** | NVIDIA | Multi-modal | 16.60% | Supports multiple languages |

## Features

- **Multi-model support**: Easy switching between different ASR architectures
- **GPU acceleration**: CUDA optimization for all supported models
- **Chunked processing**: Handles long audio files (1+ hours) efficiently
- **Format flexibility**: Supports MP3, WAV, M4A, FLAC, and other common formats
- **Comprehensive evaluation**: WER, accuracy, precision, recall, F1-score metrics
- **Automatic preprocessing**: Audio format conversion and normalization

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg for audio processing

### Dependencies
```bash
# Core dependencies
pip install torch torchaudio transformers soundfile pydub

# Model-specific dependencies
pip install nemo_toolkit  # For Parakeet and Canary
pip install moshi==0.2.11 julius librosa sphn  # For Moshi STT
```
## Usage

### Basic Transcription

```bash
# Transcribe with various models
python .src/<model_name>_transcriber.py

```

All scripts expect your audio file to be named `audioFile.wav` or `audioFile.mp3` in the current directory.

### Advanced Usage

#### Chunked Processing for Long Audio
```python
from moshi_transcriber import MoshiTranscriber

transcriber = MoshiTranscriber()
result = transcriber.transcribe_file(
    input_audio="long_podcast.mp3",
    max_chunk_duration=90.0,  # 90-second chunks
    output_file="transcript.txt",
    timestamped_output="timestamped.txt"
)
```

#### Batch Processing
```python
from parakeet_transcriber import ParakeetTranscriber

transcriber = ParakeetTranscriber()
transcriber.transcribe_directory(
    input_dir="./audio_files/",
    output_dir="./transcripts/"
)
```

### Evaluation

Calculate Word Error Rate

```bash
# Basic WER calculation
python wer_calculator.py ground_truth.txt asr_output.txt


### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"  # For transcript correction
export CUDA_VISIBLE_DEVICES="0"          # GPU selection
export HF_HOME="/path/to/huggingface"    # Model cache location
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce chunk size or batch size
   python transcriber.py --chunk-duration 30
   ```

2. **Audio Format Errors**
   - Ensure FFmpeg is installed for format conversion
   - Check audio file integrity with `ffprobe`

3. **Model Loading Errors**
   ```bash
   # Clear HuggingFace cache
   rm -rf ~/.cache/huggingface/
   ```

4. **Repetition in Granite Output**
   - Use improved generation parameters
   - Consider alternative models for production use

### Performance Optimization

- Use GPU acceleration when available
- Adjust chunk sizes based on available memory
- Enable mixed precision for faster inference
- Use SSD storage for model caching

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Add your changes with tests
4. Submit a pull request

### Adding New Models

To add support for a new ASR model:

1. Create a new transcriber class inheriting from `BaseTranscriber`
2. Implement required methods: `load_model()`, `transcribe_chunk()`, `preprocess_audio()`
3. Add evaluation tests with sample audio
4. Update benchmarking scripts

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kyutai](https://github.com/kyutai-labs/moshi) for Moshi STT
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for Parakeet and Canary
- [IBM](https://huggingface.co/ibm-granite) for Granite models
- [OpenAI](https://openai.com) for GPT-based correction capabilities

## Roadmap

- [ ] Add support for Whisper v3 Large
- [ ] Add multilingual benchmarking
- [ ] Add End-to-End Neural Architectures
- [ ] Add multilingual benchmarking