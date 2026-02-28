# MSCNN-BiLSTM-AE for Unsupervised NIDS

Two-stage unsupervised anomaly detection for network intrusion detection:

- **Stage 1**: Multi-Scale CNN Autoencoder (per-flow spatial features)
- **Stage 2**: LSTM Autoencoder (temporal patterns on latent sequences)

**Training**: Benign CIC-IDS-2017 only  
**Primary eval**: CSE-CIC-IDS-2018 (unseen)  
**Secondary eval**: CIC-IDS-2017 all-label

## Colab

```bash
# Clone di Colab
!git clone https://github.com/YOUR_USERNAME/mscnn-bilstm-ae-28-feb.git
%cd mscnn-bilstm-ae-28-feb
```

Lalu jalankan `notebooks/colab_runner.ipynb` (atur path dataset di config).

## Local

```bash
pip install -r requirements.txt
python -m src.main --config src/config.yaml
```

## Struktur

- `src/` — kode utama (data, model, training, evaluation)
- `notebooks/colab_runner.ipynb` — runner untuk Google Colab
- `src/config.yaml` — konfigurasi

Dataset: letakkan CSV CIC-IDS-2017 dan CSE-CIC-IDS-2018 di `data/raw/` sesuai path di config.
