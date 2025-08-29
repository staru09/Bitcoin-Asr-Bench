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
| **Granite** | IBM | Seq2Seq | 97.80%* | *Issues with repetition loops |

## Features

- **Multi-model support**: Easy switching between different ASR architectures
- **GPU acceleration**: CUDA optimization for all supported models
- **Chunked processing**: Handles long audio files (1+ hours) efficiently
- **Format flexibility**: Supports MP3, WAV, M4A, FLAC, and other common formats
- **Comprehensive evaluation**: WER, accuracy, precision, recall, F1-score metrics
- **Automatic preprocessing**: Audio format conversion and normalization
- **Transcript correction**: OpenAI GPT-based post-processing for improved accuracy
- **Visualization**: Performance comparison plots and dashboards

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
pip install openai  # For transcript correction

# Visualization and evaluation
pip install matplotlib seaborn numpy
```

### Quick Start with UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/asr-benchmark
cd asr-benchmark

# Install with UV package manager
uv pip install -r requirements.txt
```

## Usage

### Basic Transcription

```bash
# Transcribe with Moshi STT (best performance)
python moshi_transcriber.py

# Transcribe with NVIDIA Parakeet
python parakeet_transcriber.py

# Transcribe with NVIDIA Canary
python canary_transcriber.py
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

### Transcript Correction

Improve ASR output using OpenAI GPT models:

```bash
export OPENAI_API_KEY="your-api-key"

# Correct transcript with Bitcoin/crypto domain knowledge
python transcript_corrector.py transcript.txt -d bitcoin -o corrected.txt

# Custom corrections with keywords
python transcript_corrector.py transcript.txt -k "Bitcoin,Lightning,Satoshi" -o corrected.txt
```

### Evaluation

Calculate Word Error Rate and generate performance reports:

```bash
# Basic WER calculation
python wer_calculator.py ground_truth.txt asr_output.txt

# Detailed analysis with visualizations
python wer_calculator.py ground_truth.txt asr_output.txt --detailed-analysis -o report.txt

# Generate comparison plots
python asr_plotter.py
```

## Performance Benchmarks

Based on a 24-minute Bitcoin podcast episode:

| Model | WER | Accuracy | F1-Score | Processing Time | GPU Memory |
|-------|-----|----------|----------|----------------|------------|
| Moshi | 4.31% | 97.94% | 97.29% | ~2 min | ~8GB |
| Parakeet | 9.06% | 98.86% | 95.36% | ~3 min | ~6GB |
| Canary | 16.60% | 97.94% | 91.47% | ~4 min | ~10GB |
| Granite | 97.80% | 77.93% | 57.79% | ~5 min | ~12GB |

*Tested on NVIDIA H100 NVL with 24-minute audio file*

## Project Structure

```
asr-benchmark/
├── transcribers/
│   ├── moshi_transcriber.py      # Kyutai Moshi STT
│   ├── parakeet_transcriber.py   # NVIDIA Parakeet
│   ├── canary_transcriber.py     # NVIDIA Canary  
│   └── granite_transcriber.py    # IBM Granite (deprecated)
├── evaluation/
│   ├── wer_calculator.py         # WER computation
│   ├── transcript_corrector.py   # GPT-based correction
│   └── asr_plotter.py           # Visualization tools
├── utils/
│   └── audio_preprocessing.py    # Audio format utilities
├── examples/
│   └── sample_transcripts/       # Example outputs
├── tests/
│   └── test_transcribers.py     # Unit tests
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"  # For transcript correction
export CUDA_VISIBLE_DEVICES="0"          # GPU selection
export HF_HOME="/path/to/huggingface"    # Model cache location
```

### Model-Specific Settings

#### Moshi STT
- Chunk duration: 90 seconds (optimal)
- Overlap: 2 seconds
- Sample rate: Auto-detected
- VAD: Optional voice activity detection

#### NVIDIA Models (Parakeet/Canary)
- Chunk duration: 45 seconds
- Required format: 16kHz mono WAV
- Batch size: 1 (recommended for GPU memory)

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
