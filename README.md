# Automated White Blood Cell Classification Using CNNs

_A clinical decision-support prototype for first-pass WBC differentials from peripheral blood smear images._

## 1) Clinical motivation

Manual white blood cell (WBC) differentials are time-consuming and can be affected by fatigue, workload, and inter-observer variability—especially in high-throughput laboratory settings. This project explores a deployable deep learning pipeline that provides consistent first-pass WBC classification with a built-in human-review workflow for uncertain predictions.

Intended use (prototype): assistive triage and rapid first-pass differentials.
Not intended for: autonomous diagnosis, replacing technologist review, or use without site-specific validation.

## 2) Dataset overview (Acevedo et al.)

This project uses the Acevedo et al. labeled peripheral blood smear image dataset as the primary training and evaluation source. The dataset provides labeled examples of WBC morphologies suitable for supervised learning and model benchmarking.

High-level data handling

Images are standardized via resizing and normalization.

Class labels are mapped to a consistent taxonomy used across training, evaluation, and UI reporting.

Splits are performed to support fair evaluation (train/validation/test).

Note: If you reuse this repo, follow the dataset’s license/terms and cite the dataset appropriately in derived work.

## 3) Modeling approach

Two complementary CNN approaches are implemented to balance performance and deployability:

### A) Custom CNN (lightweight morphology model)

A compact five-block convolutional network optimized for WBC morphology:

Repeated blocks of: Conv (3×3) → ReLU → MaxPool → Dropout (0.3)

Global Average Pooling prior to classification

Final Dense Softmax for class probabilities

Why: small footprint, fast inference, easier iteration and deployment.

### B) Transfer learning (MobileNetV2)

A MobileNetV2-based classifier with transfer learning:

Pretrained convolutional backbone

Fine-tuned head for WBC classes

Regularization and controlled fine-tuning for generalization

Why: strong baseline, robust feature extraction, efficient performance on limited compute.

## 4) Evaluation strategy

Model performance is assessed using:

Held-out test evaluation

Confusion matrix to identify systematic confusions

Precision/recall by class (to understand false positives vs false negatives)

Aggregate metrics (e.g., accuracy, macro averages) where appropriate

Clinical framing: some errors are “more costly” than others. We emphasize per-class behavior and failure modes rather than relying only on overall accuracy.

## 5) Model explainability (saliency overlays)

To support transparency, the app generates saliency map overlays that highlight which pixels most influenced the prediction. This helps users sanity-check whether the model is focusing on relevant morphology rather than background artifacts.

Important: saliency methods are supportive visual aids—not proof of correctness.

## 6) Streamlit application (deployment prototype)

A Streamlit UI demonstrates how the model can be used in a practical workflow:

Key features

Upload single images or batches

Display per-image predicted class + confidence

Show WBC differential summary (class counts) for uploaded batches

Navigate images (Next/Previous) for rapid review

Display saliency overlay per image

This design supports a “human-in-the-loop” approach: confident predictions can speed review, while uncertain results remain clearly flagged for manual confirmation.

## 7) Limitations & regulatory considerations

This repository is a prototype intended for research/portfolio demonstration.

Limitations

Dataset domain shift: staining, imaging hardware, and site-specific protocols may change image appearance.

Class imbalance: rare classes may be harder to learn and evaluate reliably.

Generalization: strong performance on one dataset does not guarantee performance in production.

Regulatory/quality notes

Any clinical deployment would require: site-specific validation, QC monitoring, audit logging, and governance consistent with laboratory quality systems.

## 8) Project structure
app/                # Streamlit UI
src/                # reusable pipeline modules
models/             # trained weights (not stored if too large; see release notes)
notebooks/          # experiments and evaluation (clean, minimal)
assets/             # diagrams + screenshots for the README
requirements.txt

## 9) How to run locally
### 1) Create environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

### 2) Start the app
streamlit run app/app.py

## 10) Results snapshot

<img width="522" height="235" alt="image" src="https://github.com/user-attachments/assets/6bdedebe-00ce-40f4-b9a3-9a0cff534d2e" />
<img width="522" height="461" alt="image" src="https://github.com/user-attachments/assets/ba804680-5116-4e83-a325-4bf9c3cd1521" />

<img width="901" height="801" alt="image" src="https://github.com/user-attachments/assets/a4a53bb0-b3c7-485c-9fe9-5b07c67605cb" />
Figure 1. Saliency Mapping of a neutrophil.


## 11) Citation

Acevedo, A., Merino, A., Alférez, S., Molina, A., Boldú, L., & Rodellar, J. (2020). A dataset for microscopic peripheral blood cell classification. Mendeley Data, V1. https://data.mendeley.com/datasets/snkd93bnjr/1

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2980–2988.

Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, 42, 60–88.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.

Singh, K. (2024). Artificial intelligence in hematology: A critical perspective. Journal of Clinical Hematology and Oncology Research, 6(1), 1–6. https://www.probiologists.com/article/artificial-intelligence-in-hematology-a-critical-perspective
