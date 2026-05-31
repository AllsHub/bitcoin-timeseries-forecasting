# Bitcoin Time-Series Forecasting — LSTM Encoder–Decoder with Attention

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-Time%20Series-3F4F75)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Multivariate, multi-horizon forecasting of Bitcoin price built around a **Seq2Seq LSTM Encoder–Decoder** enhanced with a **custom Multi-Head Attention** mechanism.

## Highlights
- **Time-series EDA** — feature-correlation heatmap, ACF/PACF diagnostics, and seasonal decomposition to characterize the signal.
- **Feature engineering & scaling** — transforms raw OHLC-style data into model-ready tensors (MinMax scaling).
- **Windowing pipeline** — sliding-window generator producing multi-horizon forecast targets.
- **Architecture** — LSTM Encoder–Decoder with a custom multi-head attention layer over the encoder states.
- **Custom training loop** — fine-grained control over teacher forcing and optimization.
- **Evaluation** — learning-curve analysis and MAE on held-out horizons.

## Tech Stack
`TensorFlow` · `Keras` · `statsmodels` · `scikit-learn` · `Pandas` · `NumPy`

## Run
```bash
pip install -r requirements.txt
jupyter notebook bitcoin_forecasting.ipynb
```

## Author

**Aldo Maretra Putra** — Astronomy undergraduate & ML/AI engineer
[LinkedIn](https://linkedin.com/in/aldomrtr) · [GitHub](https://github.com/AllsHub)
