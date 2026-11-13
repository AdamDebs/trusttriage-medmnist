# TrustTriage results

### breastmnist (malignant as positive; missed positive = 5 x false alarm)


CNN: 40 epochs, best validation AUC 0.926, Platt scaling a = 1.40, b = -0.06

| Model and threshold | AUC | Brier | ECE | Threshold | Sensitivity | Specificity | Missed positives | False alarms | Cost per case |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression @0.5 | 0.828 | 0.145 | 0.089 | 0.50 | 42.9% | 93.0% | 24 | 8 | 0.821 |
| CNN @0.5 (uncalibrated) | 0.877 | 0.119 | 0.093 | 0.50 | 69.0% | 89.5% | 13 | 12 | 0.494 |
| CNN @0.5 (calibrated) | 0.877 | 0.116 | 0.073 | 0.50 | 69.0% | 91.2% | 13 | 10 | 0.481 |
| CNN @cost-optimal theory (uncalibrated) | 0.877 | 0.119 | 0.093 | 0.17 | 90.5% | 56.1% | 4 | 50 | 0.449 |
| CNN @cost-optimal theory (calibrated) | 0.877 | 0.116 | 0.073 | 0.17 | 83.3% | 70.2% | 7 | 34 | 0.442 |
| CNN @tuned on validation (calibrated) | 0.877 | 0.116 | 0.073 | 0.20 | 83.3% | 74.6% | 7 | 29 | 0.410 |

Selective prediction at 80% coverage: 31 cases deferred to a clinician; missed positives among automated decisions 4 (vs 7 if everything were automated); automated cost per case 0.296.

| Shift severity | AUC | Cost per case | Missed positives | MMD detection rate |
|---|---|---|---|---|
| 0 | 0.877 | 0.410 | 7 | 5% |
| 1 | 0.859 | 0.583 | 1 | 100% |
| 2 | 0.686 | 0.788 | 16 | 100% |
| 3 | 0.423 | 1.346 | 42 | 100% |

(Severity 0 is no shift, so its detection rate is the false-alarm rate; target about 5%.)


### pneumoniamnist (pneumonia as positive; missed positive = 5 x false alarm)


CNN: 37 epochs, best validation AUC 0.996, Platt scaling a = 1.29, b = -0.32

| Model and threshold | AUC | Brier | ECE | Threshold | Sensitivity | Specificity | Missed positives | False alarms | Cost per case |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression @0.5 | 0.927 | 0.120 | 0.136 | 0.50 | 98.2% | 62.4% | 7 | 88 | 0.197 |
| CNN @0.5 (uncalibrated) | 0.945 | 0.149 | 0.170 | 0.50 | 99.2% | 52.6% | 3 | 111 | 0.202 |
| CNN @0.5 (calibrated) | 0.945 | 0.147 | 0.161 | 0.50 | 99.0% | 53.4% | 4 | 109 | 0.207 |
| CNN @cost-optimal theory (uncalibrated) | 0.945 | 0.149 | 0.170 | 0.17 | 100.0% | 35.0% | 0 | 152 | 0.244 |
| CNN @cost-optimal theory (calibrated) | 0.945 | 0.147 | 0.161 | 0.17 | 100.0% | 42.3% | 0 | 135 | 0.216 |
| CNN @tuned on validation (calibrated) | 0.945 | 0.147 | 0.161 | 0.19 | 100.0% | 45.3% | 0 | 128 | 0.205 |

Selective prediction at 80% coverage: 124 cases deferred to a clinician; missed positives among automated decisions 0 (vs 0 if everything were automated); automated cost per case 0.138.

| Shift severity | AUC | Cost per case | Missed positives | MMD detection rate |
|---|---|---|---|---|
| 0 | 0.945 | 0.205 | 0 | 35% |
| 1 | 0.795 | 0.729 | 74 | 100% |
| 2 | 0.556 | 2.232 | 266 | 100% |
| 3 | 0.518 | 0.510 | 21 | 100% |

(Severity 0 is no shift, so its detection rate is the false-alarm rate; target about 5%.)
