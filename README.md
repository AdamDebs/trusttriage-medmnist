# TrustTriage: calibrated, cost-aware medical image triage with shift monitoring

In my second-year machine learning coursework I built a classifier for breast cancer. Lowering its decision threshold caught more cancers but raised false alarms, and I could not say how that threshold *should* be set from the cost of each error. This project is my attempt to answer that question properly, and to ask two more: when should a model hand a case to a clinician, and how can it notice that its data has changed?

I tested this on two public [MedMNIST](https://medmnist.com/) tasks: **BreastMNIST** (breast ultrasound, malignant vs normal/benign) and **PneumoniaMNIST** (chest X-ray, pneumonia vs normal).

## Method

1. **Baseline and model:** a logistic-regression baseline on pixels and a small convolutional neural network in PyTorch.
2. **Calibration:** Brier score, expected calibration error (ECE) and reliability diagrams, with Platt scaling fitted on the validation set.
3. **Cost-aware thresholds:** with a missed positive costing 5x a false alarm, I compare the default 0.5 threshold, a threshold chosen on validation data, and the theoretical cost-optimal threshold C_FP / (C_FP + C_FN).
4. **Selective prediction:** the least certain cases are deferred to a clinician (risk-coverage curve).
5. **Label-free shift monitoring:** a simulated scanner change (blur, lower contrast, noise) is detected with a kernel two-sample test (maximum mean discrepancy, MMD) on the network's embeddings.

## Results

### Breast ultrasound (BreastMNIST, 156 test images, 42 malignant)

| Model and threshold | AUC | ECE | Missed malignancies | False alarms | Cost per case |
|---|---|---|---|---|---|
| Logistic regression, 0.5 | 0.828 | 0.089 | 24 | 8 | 0.821 |
| CNN, 0.5 | 0.877 | 0.093 | 13 | 12 | 0.494 |
| CNN calibrated, theoretical threshold 0.17 | 0.877 | 0.073 | 7 | 34 | 0.442 |
| **CNN calibrated, threshold 0.20 chosen on validation** | **0.877** | **0.073** | **7** | **29** | **0.410** |

![BreastMNIST results](results/breastmnist/summary.png)

### Chest X-ray (PneumoniaMNIST, 624 test images)

| Model and threshold | AUC | ECE | Missed pneumonia | False alarms | Cost per case |
|---|---|---|---|---|---|
| Logistic regression, 0.5 | 0.927 | 0.136 | 7 | 88 | 0.197 |
| CNN, 0.5 | 0.945 | 0.170 | 3 | 111 | 0.202 |
| **CNN calibrated, threshold 0.19 chosen on validation** | **0.945** | **0.161** | **0** | **128** | **0.205** |

![PneumoniaMNIST results](results/pneumoniamnist/summary.png)

Full tables, including selective prediction and the shift experiments, are in [`results/summary.md`](results/summary.md); every number is in [`results/metrics.json`](results/metrics.json).

## Findings

* **Cost-aware thresholds work.** On breast ultrasound, calibrating the CNN and choosing the threshold on validation data cut missed malignancies from 13 to 7 of 42 (sensitivity 69.0% to 83.3%), at the cost of 17 more false alarms, and lowered cost per case from 0.481 to 0.410. The theoretical cost-optimal threshold came close (0.442).
* **Deferral helps.** Sending the least certain 20% of breast cases to a clinician cut missed malignancies among automated decisions from 7 to 4.
* **Monitoring catches shift early.** The MMD test flagged every batch at the mildest simulated shift, where AUC barely moved (0.877 to 0.859) but cost per case rose by 42%, with a 5% false-alarm rate on clean data.
* **Validation can mislead.** On PneumoniaMNIST, validation AUC was 0.996 but test AUC was 0.945, and the MMD test flagged 35% of clean test batches against the validation data. This suggests the test images differ from the validation images, and it explains why calibration fitted on validation barely helped on the test set. On cost, the CNN matched the logistic-regression baseline without beating it.

## Limitations

* Images are 28 x 28 pixels, far below clinical resolution.
* BreastMNIST's validation set is small (78 images), so calibration and thresholds fitted on it are noisy.
* The shift is simulated; a real scanner or hospital change may look different. MMD detects a change in the data, not a drop in accuracy.
* The 5:1 cost ratio is an illustrative assumption, not a clinical value.
* This is a learning and research project, not a medical device.

## Run it

**Google Colab:** open `TrustTriage.ipynb` in Colab and choose `Runtime > Run all` (a few minutes on CPU; the data downloads automatically).

**Locally:**
```bash
pip install -r requirements.txt
python trusttriage.py
```

## Repository

| File | Contents |
|---|---|
| `TrustTriage.ipynb` | Notebook with all code and outputs |
| `trusttriage.py` | The same pipeline as a script |
| `results/` | Tables, metrics and figures from my run |

## Data and citation

Data from MedMNIST: J. Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification", *Scientific Data* 10, 41 (2023).

---

*AI tools were used throughout parts of this project to assist with coding, debugging, and understanding errors. I independently worked through the project, made the design and implementation decisions, and used AI as a support tool when needed.*
