# v0.22 Vision Literature Extension — 2026-09-10

This note separates external research from the uploaded ICT/TTrades source material. It motivates future challengers; it does not modify the frozen v0.22b acceptance gates.

## Findings relevant to our design

- Financial time-series-to-image research has used Gramian Angular Fields (GASF/GADF), recurrence plots and multi-resolution CNNs. These representations are useful challengers because they encode pairwise/temporal geometry differently from ordinary candlestick rasters.
- A 2026 Expert Systems with Applications study applies GADF + CNN to high-frequency Bitcoin pattern recognition and then clusters model relevance maps, supporting an interpretability track rather than treating a black-box image score as sufficient evidence.
- A 2026 Discover Computing study combines GAF, a dual-branch CNN, BiLSTM and attention, supporting our planned separation of visual and temporal encoders followed by fusion.
- A 2025 candlestick-image CNN study reports that explicit candlestick-pattern detection did not improve its image-only model, which reinforces the need for ablation rather than assuming named patterns add predictive information.
- An April 2026 preprint systematically compares raw candlestick images, GAF and multi-channel GAF for cryptocurrency regime prediction and reports strong results for a relatively simple raw-candlestick CNN. Because this is a preprint, it is treated as a challenger hypothesis, not authoritative evidence.

## Consequence for the project

After v0.22b, the next visual challenger set should include raw candles, GASF, GADF, recurrence plots and multi-resolution combinations. These should be compared under the same chronological split and with the same leakage audit. Only representation improvements that survive validation and untouched test should enter multimodal fusion.

## Public references

- Distler, Okhrin & Pfahler, *A spectral relevance analysis approach to pattern recognition of financial time series*, Expert Systems with Applications, 2026, DOI 10.1016/j.eswa.2025.129555.
- Zhang & Chang, *Stock price prediction based on GAF and enhanced dual-branch CNN model*, Discover Computing, 2026.
- Barra et al., *Deep Learning and Time Series-to-Image Encoding for Financial Forecasting*, IEEE/CAA Journal of Automatica Sinica, 2020, DOI 10.1109/JAS.2020.1003132.
- Duong et al., *Investigating Market Strength Prediction with CNNs on Candlestick Chart Images*, arXiv:2501.12239, 2025.
- Haggett, *Visual Chart Representations for Cryptocurrency Regime Prediction: A Systematic Deep Learning Study*, arXiv:2605.00875, 2026 — preprint.
