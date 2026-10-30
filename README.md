# Network Intrusion Detection – GDG on Campus ML Challenge

4-class classification of network flows (**Benign, Availability, Reconnaissance, Spoofing**), scored by **Macro F1** on Kaggle.

- **Final public leaderboard Macro F1:** 0.84811
- **Libraries:** scikit-learn only (no GPU, runs in about 1 minute)
- **Seed:** 42 (fully reproducible; re-running gives an identical `submission.csv`)

## Data
- `train.csv`: 84,713 flows × 84 columns (82 features, `id`, `Target`)
- `test.csv`: 33,161 flows
- Class counts in train: Availability 80,072 · Spoofing 2,323 · Benign 1,583 · Reconnaissance 735

The data is heavily imbalanced. Availability makes up about 95% of the training rows, and the other three classes together have only 4,641 rows.

## Approach

### 1. Key finding: destination port 21 identifies Availability
In train, 79,698 of the 79,699 rows with `id.resp_p == 21` are Availability, and the one exception is Reconnaissance. Port 21 therefore gives an almost perfect rule for most of the Availability class, so I predict Availability for every row with `id.resp_p == 21`.

### 2. Why not let a model learn Availability from flow features?
There is a train/test distribution shift on reply traffic:

| | share of flows with `bwd_pkts_tot == 0` (no reply) |
|---|---|
| Train Availability flows | 0.1% |
| Test port-21 flows | 88.1% |

A model that learns "Availability = flows that get replies" from train would reject most test port-21 flows and label them Reconnaissance or Benign. The port is a more stable signal than the flow-level reply statistics, so the port rule decides Availability.

### 3. Model for the other three classes
A 3-class model is trained **only on Benign / Reconnaissance / Spoofing rows** (4,641 rows). The 80k Availability rows are left out, so they don't dominate training or the class balance.

Two models are averaged by summing their predicted probabilities:
- `HistGradientBoostingClassifier` (250 iterations, learning rate 0.06, `class_weight="balanced"`, native categorical handling for `proto` and `service`)
- `ExtraTreesClassifier` (300 trees, `class_weight="balanced"`, categories as integer codes)

Why this pair:
- `class_weight="balanced"` matters because Macro F1 gives each class equal weight. Reconnaissance, the smallest class, counts as much as the rest.
- A boosted model and a bagged random-split model make different kinds of errors, so averaging them gives more stable predictions than either alone.
- Both are fast on a few thousand rows and need no heavy tuning.

Categorical encoding: the vocabulary of `proto` and `service` is built from train and test together, so categories that only appear in test are handled consistently.

### Feature engineering
I did **not** create new features. The model uses all 82 raw flow features as given (ports, protocol, service, packet and byte counts, flag counts, inter-arrival times, window sizes). The feature work was about which columns to trust:
- `proto` and `service` are categorical. HistGradientBoosting uses them natively, and ExtraTrees gets integer codes. The vocabulary is built from train and test together.
- Missing values are filled with -1 for ExtraTrees. HistGradientBoosting handles them itself.
- Flow features such as `bwd_pkts_tot` are deliberately **not** used to decide Availability, because of the train/test shift shown above. Only the destination port (`id.resp_p`) is used for that class.

### 4. Final prediction
1. Take the argmax of the averaged 3-class probabilities.
2. Overwrite every row with `id.resp_p == 21` as **Availability**.

Predicted class shares on test: Benign 31.8%, Availability 31.4%, Reconnaissance 20.8%, Spoofing 16.1%.

## Evaluation methodology
- **Leaderboard:** the Kaggle public score is Macro F1 **0.84811**.
- **Local check:** `cv_evaluation.py` runs 5-fold stratified cross-validation (seed 42) on the 4,641 non-Availability training rows, which is the part the models handle. Availability is decided by the port rule, so it isn't part of this check.

| Model | CV Macro F1 (3 classes) |
|---|---|
| HistGradientBoosting | 0.9908 |
| ExtraTrees | 0.9923 |
| Ensemble (submitted) | 0.9913 |

Per class (ensemble): Benign F1 0.9876, Reconnaissance 0.9946, Spoofing 0.9916.

**How to read the gap between 0.99 (CV) and 0.848 (leaderboard):** random folds from train are an optimistic estimate, because train and test come from different distributions (see the reply-traffic shift above). Cross-validation on train cannot measure that shift, so the leaderboard is the more honest number. I chose the ensemble because it averages two different model types and its CV score is within 0.001 of the best single model, not because it won the CV.

## Limitations
- The port-21 rule is a data-driven rule, not something the model learned. It works because of how this dataset was generated, and it would not generalize to real network traffic.
- Availability traffic on other ports would be missed. Train has no such cases, so I couldn't test this.
- Cross-validation on train is much higher than the leaderboard score, so I don't trust it as a measure of test performance. The cause is the distribution shift, and I didn't find a way to fix it.

## How to run
```bash
pip install -r requirements.txt
mkdir data && cp /path/to/{train,test,sample_submission}.csv data/
python solution.py          # writes submission.csv
```
The script finds the data folder itself (it searches `/kaggle/input` and the current directory) and checks the output against `sample_submission.csv` (columns, IDs, order). On Kaggle it writes to `/kaggle/working/submission.csv`.

## Files
- `solution.py`: full pipeline (the final code used for the submission)
- `cv_evaluation.py`: 5-fold cross-validation of the 3-class model
- `submission.csv`: predictions submitted to Kaggle
- `requirements.txt`: dependencies
- Data files are not included, so get them from the Kaggle challenge page.
