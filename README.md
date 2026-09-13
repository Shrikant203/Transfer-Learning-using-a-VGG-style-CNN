# Task 2 — Transfer Learning using a VGG-style CNN

Demonstrates the full transfer-learning workflow: pretrain a VGG-style CNN
(from scratch, NumPy) on a source domain, freeze the convolutional base,
train a new classifier head on a small, domain-shifted target dataset,
fine-tune the last convolutional block, and compare against an identical
model trained from scratch (no pretraining).

## Files
- `Transfer_Learning.ipynb` — full notebook, already executed, with all outputs and plots. Includes an Appendix with a runnable reference implementation using a real ImageNet-pretrained VGG16 (TensorFlow/Keras).
- `Transfer_Learning_Report.pdf` — full report: objective, architecture, methodology, source code, console output, figures, results comparison, observations.
- `models/pretrained_source_model.pkl` — the CNN after pretraining on the source domain.
- `models/transfer_learning_model.pkl` — the model after freezing + fine-tuning on the target domain.
- `models/baseline_model.pkl` — the same architecture trained from scratch on the target domain only (for comparison).

## Result :
Transfer learning: **48.57% test accuracy** vs baseline (from scratch):
**38.57%** — a **+10 percentage point** improvement, with substantially
better generalization (lower test loss: 1.41 vs 2.98).

## Running :
Open `Transfer_Learning.ipynb` in Jupyter, VS Code, or Google Colab,
and run all cells top-to-bottom. Requires `numpy`, `matplotlib`, `seaborn`,
`scikit-learn`, and `Pillow`. The Appendix A reference cell additionally
requires `tensorflow` and internet access (only needed if you choose to run
that specific cell).
