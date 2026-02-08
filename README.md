# 🎸 Open-Fret: MIDI to Guitar Tablature Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**Open-Fret** is a high-performance, research-driven MIDI to Guitar Tablature transcription engine. Based on the **Fretting-Transformer** architecture, it leverages a custom-tailored T5 Transformer model to translate symbolic music (MIDI) into ergonomic, playable, and accurate guitar tabs.

---

## ✨ Key Features

- **🎯 High Accuracy**: Translates MIDI notes into precise string and fret combinations.
- **⚡ CPU Optimized**: Specifically designed `T5-Tiny-Tab` architecture for fast inference on standard hardware.
- **🎸 Ergonomics First**: Learns playable fingerings from massive datasets of real GuitarPro files.
- **🔧 Multi-Config**: Support for custom tunings and capo positions (work in progress).
- **📦 End-to-End Pipeline**: Complete scripts for data downloading, processing, vocabulary generation, and training.

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/open-fret.git
cd open-fret
pip install -r requirements.txt
```

### 2. Dataset Preparation

Open-Fret uses the **DadaGP** and **SynthTab** datasets. To prepare your data, follow the sequential scripts in the `scripts/` directory:

```bash
# 1. Download the dataset (if applicable/accessible)
python scripts/00_download.py

# 2. Process MIDI/GP files into raw token streams
python scripts/01_process.py

# 3. Generate the SentencePiece vocabulary
python scripts/01.5_generate_vocab.py

# 4. Tokenize the sequences for the model
python scripts/02_tokenize.py

# 5. Split data into train, validation, and test sets
python scripts/03_split.py

# 6. Convert predicted tokens back to readable tablature (.txt)
python scripts/tokens_to_tab.py --input path/to/predictions.tokens
```

### 3. Training

Train the `T5-Tiny-Tab` model using your prepared data:

```bash
python scripts/train_model.py
```

*Note: Optimized to train on GPUs with as little as 8GB VRAM using the Adafactor optimizer.*

### 4. Inference

Convert your MIDI files to tokens, and then to tablature:

```bash
python scripts/inference.py --input path/to/your/song.mid
```

---

## 🧠 Architecture: T5-Tiny-Tab

The core of Open-Fret is a compact T5 (Text-to-Text Transfer Transformer) model tailored for symbolic music translation:

- **Type**: Encoder-Decoder (Seq2Seq)
- **Layers**: 6 Encoder / 6 Decoder (Optimized configuration)
- **Dimensions**: `d_model = 256`, `d_ff = 2048`
- **Attention**: 8 Heads
- **Tokenizer**: Custom SentencePiece (MIDI + Tab combined vocab)

### Training Highlights
- **Optimizer**: Adafactor (Memory efficient)
- **Precision**: FP16 (Mixed Precision)
- **Batch Size**: 16 (with gradient accumulation)
- **Hardware**: Optimized for 8GB+ VRAM GPUs, but extremely fast on CPU for inference.

---

## 📊 Datasets

Open-Fret is trained on:
- **[DadaGP](https://github.com/dada-bots/dadaGP)**: A massive symbolic dataset of ~26k GuitarPro songs.
- **[SynthTab](https://github.com/yongyizang/SynthTab)**: Large-scale synthesized guitar tablature.

---

## 🗺️ Roadmap

- [ ] Support for explicit technique tokens (Bends, Slides, Vibrato).
- [ ] Integration with audio transcription (Audio → MIDI → Tab).
- [ ] Interactive Web Dashboard for music visualization.
- [ ] ONNX export for even faster CPU deployment.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Primary Research**: [Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription](https://arxiv.org/abs/2506.14223) by Anna Hamberger, Sebastian Murgul, Jochen Schmidt, and Michael Heizmann.
- Built with ❤️ using [HuggingFace Transformers](https://huggingface.co/transformers/).
