# dl-proj2

Graph neural network training code for atomic charge prediction.

## Repository Layout

- GitHub repository: code, training scripts, configs, and docs.
- Hugging Face Hub: published model artifacts (weights/config).

## Hugging Face Model Artifacts

Model repo:
https://huggingface.co/kchek2546/UNGdotEdu1

Direct links:
- Weights (.pt): https://huggingface.co/kchek2546/UNGdotEdu1/resolve/main/model_final_best.pt
- Config (.json): https://huggingface.co/kchek2546/UNGdotEdu1/resolve/main/model_final_best.config.json

## Upload Commands

```bash
hf upload kchek2546/UNGdotEdu1 trained-model/model_final_best.pt model_final_best.pt
hf upload kchek2546/UNGdotEdu1 trained-model/model_final_best.config.json model_final_best.config.json
```

## Training Notes

Operational notes and cluster commands are in docs/readme.md and docs/vibe-code-notes.md.
