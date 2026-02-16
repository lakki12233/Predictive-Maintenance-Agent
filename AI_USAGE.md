# AI Usage Documentation

## Summary
This project used AI tools (GitHub Copilot, ChatGPT) for code generation and documentation. All AI outputs were tested and verified by the developer.

---

## What AI Helped With

| Component | AI Contribution | Human Verification |
|-----------|-----------------|-------------------|
| **Data Pipeline** | Generated data loading scripts, feature engineering code | Tested on sample data, validated calculations |
| **Sensor Model** | Generated LightGBM training script, metrics code | Cross-validated, achieved AUC 0.998 |
| **Image Model (MobileNetV3)** | Generated ONNX inference wrapper, training code in Colab | Tested on 10 images, achieved 100% accuracy |
| **Image Model (CLIP)** | Generated VLM inference wrapper, CLIP integration | Tested on 10 images, found 70% accuracy with false positives |
| **Model Comparison** | Generated comparative evaluation script | Ran 10-image test suite, documented results |
| **Environmental** | Generated rule-based risk scoring | Unit tested with 3 scenarios |
| **FastAPI Service** | Generated endpoints, schemas, error handling, model switching | Manual + load testing (100 req/s verified) |
| **Docker** | Generated Dockerfile, build commands | Built and tested container |
| **Documentation** | Generated README structure, examples, model comparison docs | Reviewed and tested all commands |

---

## Verification Methods

1. **Code Testing**: Ran all generated code, fixed bugs
2. **Model Validation**: Train/test split, manual inspection
3. **Integration Testing**: End-to-end API tests
4. **Load Testing**: 50-200 request stress tests
5. **Command Testing**: Ran every documented command

---

## Issues Found & Fixed

| Issue | Fix |
|-------|-----|
| NumPy axis parameter wrong | Fixed manually |
| ONNX preprocessing missing color conversion | Added RGB→BGR |
| CLIP model severe false positives | Evaluated both models, chose MobileNetV3 as default |
| Environmental multiplier unbounded | Added 0.5-2.0 clamping |
| NumPy 2.0 compatibility | Downgraded to 1.26.4 |
| Model switching logic needed | Added RUST_MODEL_TYPE env variable support |

---

## What AI Did NOT Do

- Final architecture decisions (human judgment)
- Threshold tuning (domain knowledge)
- Dataset selection (research required)
- Error analysis (manual inspection)
- Business logic design (fault type rules)
- Model selection decision (MobileNetV3 vs CLIP - required real testing)
- Performance benchmarking (measured actual latencies, compared models)

---

## Contribution Split

| Area | AI | Human |
|------|----|----|
| Initial code generation | 70-80% | 20-30% |
| Documentation | 50% | 50% |
| Debugging & fixes | 30% | 70% |
| Testing & verification | 10% | 90% |
| Architecture decisions | 5% | 95% |

**Net Effect**: AI accelerated development ~3-4x but required continuous human verification.

---

## Tools Used

- **GitHub Copilot (Claude Sonnet 4.5)**: Code generation, debugging, refactoring, model comparison
- **ChatGPT**: Research (LightGBM tuning, Docker best practices, VLM options)
- **Google Colab**: Training environment for both MobileNetV3 and CLIP models

---


