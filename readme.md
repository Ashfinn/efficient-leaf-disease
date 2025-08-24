# CPU-Constrained Evaluation of Modern Hybrid Transformers and CNN Architectures for Tomato Leaf Disease Classification Using the PlantVillage Datase 🌱

This project evaluates **modern CNNs and hybrid Transformer architectures** — **ResNet-50**, **ConvNeXt-Tiny**, and **FastViT-T8** — for **tomato leaf disease classification** on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

The focus is on **CPU-constrained environments** to make plant disease detection feasible in **resource-limited regions** where GPU access is rare.

---

## 📌 Features

* **Dataset**: Tomato subset of PlantVillage (16,012 images, 10 classes).
* **Architectures**: ResNet-50, ConvNeXt-Tiny, FastViT-T8 (via [timm](https://github.com/huggingface/pytorch-image-models)).
* **Training**:

  * Two-phase transfer learning (frozen backbone → full fine-tuning)
  * Mixed-precision support (AMP)
  * Early stopping & checkpointing
* **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix, inference latency.
* **Deployment**: Designed for CPU-only training & inference.

---

## 🚀 Results (Validation)

| Model         | Params (M) | Accuracy   | Weighted F1 | Latency (s/img) |
| ------------- | ---------- | ---------- | ----------- | --------------- |
| ResNet-50     | 25.6       | 99.50%     | 0.995       | 0.0352          |
| ConvNeXt-Tiny | 29.0       | **99.88%** | **0.9988**  | 0.0508          |
| FastViT-T8    | 27.4       | 99.66%     | 0.997       | **0.0219**      |

* **ConvNeXt-Tiny** → Highest accuracy
* **FastViT-T8** → Fastest inference
* **ResNet-50** → Strong classical baseline

---

## 📂 Repository Structure

```
efficient-leaf-disease/
│── datasets/               # PlantVillage (tomato subset)
│── src/                    # Training & evaluation scripts
│   └── train.py            # Main training loop (works for all models)
│── trained_models/         # Saved checkpoints (.pth)
│── evaluation_results/     # Metrics JSON + confusion matrices
│── logs/                   # Training logs (CSV)
│── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/efficient-leaf-disease.git
cd efficient-leaf-disease
pip install -r requirements.txt
```

**requirements.txt**

```
torch
torchvision
timm
scikit-learn
matplotlib
seaborn
tqdm
numpy
```

---

## 🖥️ Usage

### Training a model

```bash
python src/train.py --model resnet50
python src/train.py --model convnext_tiny
python src/train.py --model fastvit_t8
```

### Evaluating best checkpoint

After training, evaluation metrics and confusion matrix will be saved in `evaluation_results/`.

Example:

```bash
cat evaluation_results/resnet50_metrics.json
```

---

## 📊 Outputs

* `evaluation_results/<model>_metrics.json` → Accuracy, F1, latency, per-class report
* `evaluation_results/<model>_metrics.png` → Confusion matrix plot
* `trained_models/` → All checkpoints (`last.pth`, `best.pth`, per-epoch `.pth`)
* `logs/<model>.csv` → Training logs (loss & accuracy per epoch)

---

## 🌱 Applications

* Edge & mobile AI for agriculture
* Real-time tomato disease detection
* Low-resource decision support for farmers

---

## 📖 Citation

If you use this work, please cite:
**Rahman, O., et al. (2025)**
*CPU-Constrained Evaluation of Modern Hybrid Transformers and CNN Architectures for Tomato Leaf Disease Classification Using the PlantVillage Dataset.*