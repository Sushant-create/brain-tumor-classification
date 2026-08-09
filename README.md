# Brain Tumor Classification using CNN — MRI Images

A CNN that classifies brain MRI scans into four categories — glioma, meningioma,
pituitary tumor, or no tumor — served through a Flask web app with a drag-and-drop
dashboard.

## Objective
Classify brain MRI images into 4 categories (Glioma, Meningioma, No Tumor,
Pituitary) using a CNN, with a target test accuracy of 90%.

## Dataset Link
[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- 5,600 training images / 1,600 testing images
- 4 balanced classes: 1,400 training + 400 testing images per class
- Grayscale MRI slices, original size 512×512, resized to 64×64 for training

The dataset is **not** committed to this repo (see `.gitignore`) — download it
via Kaggle (or `opendatasets`, as in `notebooks/Cancer_Classification.ipynb`)
into `./brain-tumor-mri-dataset/` before running `train.py`.

## Libraries Used
- **TensorFlow / Keras** — model definition, training, inference
- **Flask** — web app backend and `/predict` API
- **NumPy** — array/image tensor handling
- **Pillow (PIL)** — image decoding and preprocessing for uploads
- **scikit-learn** — classification report, confusion matrix
- **Matplotlib / Seaborn** — training curves, confusion matrix heatmap
- **opendatasets** — Kaggle dataset download (notebook only)
- **pytest** — automated test suite

## Methodology
1. **Data understanding** — inspected class balance (1,400/400 images per
   class) and confirmed images are single-channel grayscale, 512×512 originally.
2. **Preprocessing** — resized to 64×64, rescaled pixel values to [0, 1],
   loaded via `ImageDataGenerator` in grayscale mode. Mild augmentation on the
   training set only (±10° rotation, 5% width/height shift, horizontal flip,
   5% zoom) to reduce overfitting without distorting diagnostic features.
3. **Model development** — a 3-block CNN (see architecture below) with
   `GlobalAveragePooling2D` instead of `Flatten` to keep the parameter count
   low (~305K total) and reduce overfitting risk on a relatively small dataset.
4. **Training** — Adam optimizer, sparse categorical cross-entropy loss, 40
   epochs max with `ReduceLROnPlateau` (halves LR after 4 stagnant epochs) and
   `EarlyStopping` (patience 12 on validation accuracy, restores best weights).
5. **Evaluation** — test accuracy/loss, per-class precision/recall/F1 via
   `classification_report`, and a confusion matrix heatmap.

## Model Architecture
Input: 64×64×1 grayscale image

| Block | Layers |
|---|---|
| Conv Block 1 | Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25) |
| Conv Block 2 | Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25) |
| Conv Block 3 | Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25) |
| Head | GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(4, softmax) |

- **Total parameters:** 305,252 (304,356 trainable)
- **Optimizer:** Adam · **Loss:** sparse categorical cross-entropy
- Defined once in `src/model_architecture.py` and imported by both
  `train.py` and `main.py`, so the architecture used for training always
  matches the architecture used to load saved weights.

## Results
From the reference training run in `notebooks/Cancer_Classification.ipynb`:

| Metric | Value |
|---|---|
| Final training accuracy | 98.36% |
| Best validation accuracy | 91.75% |
| **Test accuracy** | **91.75%** (target: 90%) |
| Test loss | 0.7496 |
| Train/val accuracy gap | 6.61% (mild overfitting, still above target) |

Per-class performance (test set, 400 images/class):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 0.99 | 0.76 | 0.86 |
| Meningioma | 0.93 | 0.91 | 0.92 |
| No Tumor | 0.88 | 1.00 | 0.93 |
| Pituitary | 0.89 | 1.00 | 0.94 |

Glioma has the lowest recall (0.76) — it's the class most often confused with
the others, worth a closer look at the confusion matrix in the notebook if
extending this project.

## Conclusion
The CNN reaches 91.75% test accuracy, above the 90% target, with a
lightweight architecture (~305K parameters) that trains in well under an
hour on CPU thanks to the small 64×64 input size and `GlobalAveragePooling2D`
head. The main weak point is glioma recall, suggesting deeper
augmentation or a slightly larger input resolution as the next improvement
to try. The trained model is wrapped in a Flask app (`main.py`) with a
dashboard (`templates/index.html`) so a new MRI slice can be classified
through the browser with per-class confidence scores.

**⚠️ Disclaimer:** This is a research/educational prototype, not a medical
diagnostic device. It has not been validated for clinical use.

---

## Project Structure
```
├── main.py                      # Flask app: dashboard + /predict API
├── train.py                     # Training script (CLI, configurable data dir)
├── launch.py                    # Cross-platform launcher (installs deps, starts app, opens browser)
├── requirements.txt
├── src/
│   ├── model_architecture.py    # build_model(), shared by train.py and main.py
│   └── preprocess.py            # Image preprocessing shared logic
├── templates/
│   └── index.html               # Upload dashboard UI
├── model/                       # Trained model saved here (git-ignored)
├── test_samples/
│   └── generate_samples.py      # Synthetic images for pipeline tests (not real MRI data)
├── tests/
│   ├── test_preprocess.py
│   └── test_main.py
├── notebooks/
│   └── Cancer_Classification.ipynb  # Original exploratory/training notebook
├── .gitignore
└── .gitattributes
```

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
Download the [dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
so you have `./brain-tumor-mri-dataset/Training/` and `.../Testing/`, then:
```bash
python train.py --data-dir ./brain-tumor-mri-dataset --epochs 40
```
This saves the trained model to `model/tumor_classifier.keras`.

### 3. Run the app
```bash
python main.py
```
Or use the one-command launcher, which installs missing dependencies, starts
the server, and opens the dashboard in your browser automatically:
```bash
python launch.py
```
Visit `http://localhost:5000`.

### 4. Run tests
```bash
python test_samples/generate_samples.py   # one-time: creates synthetic fixtures
pytest tests/ -v
```
The test suite covers image preprocessing (shape/dtype/range checks, grayscale
conversion, corrupt/empty file handling) and the Flask API (valid predictions,
missing file, wrong extension, corrupt image, model-unavailable case) — all
15 tests pass without needing the full Kaggle dataset or a trained model,
using a mocked model for the API tests.

## Known Limitations
- The Kaggle dataset (~5,600+1,600 images) isn't bundled with this repo — you
  need to download it yourself before training.
- No trained model weights are committed; `model/` starts empty. Run
  `train.py` (or copy in your own `.keras` file) before using `/predict`.
- Glioma recall (0.76) is the weakest spot in the current model — see
  Conclusion above.
