Reproducible implementation of **Private Aggregation of Teacher Ensembles (PATE)** for multi-class intrusion detection on the RT-IoT2022 dataset.  
This repository systematically evaluates **15 different teacher–student architectural configurations** using the classic Laplace noisy-max mechanism and a rigorous **moments accountant** (data-dependent + data-independent privacy analysis).

## Features

- Full PATE pipeline (teacher training → noisy aggregation → student training)
- Moments accountant with both **data-dependent** and **data-independent** ε
- 15 carefully designed teacher/student architecture combinations
- Support for LSTM, GRU, CNN, and MLP models
- Stratified data splitting, optional oversampling, early stopping
- Automatic CSV export of all results
- Built-in unit tests for privacy accounting correctness

## Supported Configurations

| #  | Teachers                                      | Student |
|----|-----------------------------------------------|---------|
| 1  | 1 × LSTM                                      | GRU     |
| 2  | 3×LSTM + 3×CNN + 3×MLP                        | LSTM    |
| 3  | 3×LSTM + 3×MLP + 3×CNN                        | GRU     |
| 4  | 3 × LSTM                                      | LSTM    |
| 5  | 1×LSTM + 1×GRU + 1×CNN                        | MLP     |
| 6  | 3 × MLP                                       | MLP     |
| 7  | 3 × CNN                                       | CNN     |
| 8  | 1×MLP + 1×CNN + 1×GRU + 1×LSTM                | CNN     |
| 9  | 1 × LSTM                                      | LSTM    |
| 10 | 6 × LSTM                                      | LSTM    |
| 11 | 6×CNN + 6×GRU + 6×LSTM                        | CNN     |
| 12 | 3×LSTM + 3×GRU + 3×CNN                        | MLP     |
| 13 | 3×LSTM + 4×MLP + 3×CNN                        | GRU     |
| 14 | 10 × CNN                                      | CNN     |
| 15 | 1×LSTM + 1×MLP + 1×CNN                        | GRU     |

## Requirements

```bash
Python >= 3.8
torch >= 1.12
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
Install dependencies:
Bashpip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
Dataset Setup

Download the RT-IoT2022 dataset.
Place the file at:

text/content/drive/MyDrive/RT_IOT2022.csv
(or update the DATA_PATH variable in the script).
The code automatically filters the following 12 attack classes:
textMQTT_Publish, Thing_Speak, Wipro_bulb, ARP_poisioning,
DDOS_Slowloris, DOS_SYN_Hping, Metasploit_Brute_Force_SSH,
NMAP_FIN_SCAN, NMAP_OS_DETECTION, NMAP_TCP_scan,
NMAP_UDP_SCAN, NMAP_XMAS_TREE_SCAN
How to Run
Google Colab (Recommended)

Mount Google Drive.
Upload / open the notebook or .py file.
Ensure the dataset path is correct.
Run all cells.

Local Machine
Bashpython pateexperiments.py
The script will:

Run privacy accounting unit tests
Train all 15 configurations sequentially
Save results to ./pate_results/all_configurations_summary.csv

Key Hyperparameters
ParameterValue
Batch size128
Teacher
epochs23
Student
epochs30
Learning rate1e-4
Early stopping patience3
Laplace inverse scale (β)1.3δ1e-5
Max moment order32
Teacher pool fraction80%
Public data fraction5%
Final test fraction15%

parameterValueBatch size128
Teacher epochs23S
tudent epochs30
Learning rate1e-4
Early stopping patience3
Laplace inverse scale (β)1.3δ1e-5
Max moment order32
Teacher pool fraction
80%
Public data fraction5%
Final test fraction15%
Output
After execution you will obtain a CSV file containing:

Configuration name
Data-dependent ε
Data-independent ε
Accuracy, Precision, Recall, F1-score
AUC and MAE
Mean vote margin

Privacy Accounting Notes

Uses the classic Laplace noisy-max mechanism.
Implements both data-dependent and data-independent moments accountant.
Enforces the invariant:
data_dependent_ε ≤ data_independent_ε
Includes unit tests that verify the correctness of the q-estimator and the moments calculation.

Reproducibility

Fixed random seed (SEED = 42)
Stratified splits
Deterministic teacher partitioning
Early stopping with model checkpointing

Citation
If you use this code, please cite the original PATE papers:

Papernot et al., “Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data”, ICLR 2017.
Papernot et al., “Scalable Private Learning with PATE”, ICLR 2018.


Privacy accounting code
# =====================================================================
# PATE PRIVATE ACCOUNTING AND VOTING MECHANISMS
# =====================================================================

def teacher_predictions_to_counts(teacher_preds, num_classes):
    teacher_preds = np.asarray(teacher_preds, dtype=np.int64)
    num_teachers, num_queries = teacher_preds.shape
    counts = np.zeros((num_queries, num_classes), dtype=np.int64)
    for query_index in range(num_queries):
        counts[query_index] = np.bincount(teacher_preds[:, query_index], minlength=num_classes)
    return counts

def aggregate_teacher_votes_laplace(teacher_preds, num_classes, laplace_inverse_scale, random_seed=None):
    counts = teacher_predictions_to_counts(teacher_preds, num_classes)
    rng = np.random.default_rng(random_seed)
    laplace_scale = 1.0 / laplace_inverse_scale
    noise = rng.laplace(loc=0.0, scale=laplace_scale, size=counts.shape)
    noisy_counts = counts.astype(np.float64) + noise
    released_labels = np.argmax(noisy_counts, axis=1)
    return released_labels, counts, noisy_counts

def logaddexp_scalar(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))

def logmgf_exact(q, mechanism_eps, moment_order):
    """Data-dependent log moment generating function."""
    lam = float(moment_order)
    eps = float(mechanism_eps)
    bound_1 = 0.5 * (eps ** 2) * lam * (lam + 1.0)
    bound_3 = eps * lam
    bound_2 = math.inf
    denominator = 1.0 - math.exp(eps) * q
    if q < 0.5 and denominator > 0.0:
        log_one_minus_q = math.log1p(-q)
        log_term_1 = log_one_minus_q + lam * (log_one_minus_q - math.log(denominator))
        log_term_2 = math.log(q) + eps * lam if q > 0.0 else -math.inf
        bound_2 = logaddexp_scalar(log_term_1, log_term_2)
    return min(bound_1, bound_2, bound_3)

def compute_q_noisy_max(counts, laplace_inverse_scale):
    """Estimate the probability that the noisy-max does not return the true plurality."""
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = len(counts)
    if num_classes < 2:
        return 0.0
    winner = int(np.argmax(counts))
    winning_votes = counts[winner]
    competitors = np.delete(counts, winner)
    gaps = laplace_inverse_scale * (winning_votes - competitors)
    q = float(np.sum((gaps + 2.0) * np.exp(-gaps) / 4.0))
    return min(max(q, 0.0), 1.0 - 1.0 / num_classes)

def data_dependent_log_moment(counts, laplace_inverse_scale, moment_order):
    q = compute_q_noisy_max(counts, laplace_inverse_scale)
    mechanism_eps = 2.0 * laplace_inverse_scale
    return logmgf_exact(q, mechanism_eps, moment_order)

def data_independent_log_moment(laplace_inverse_scale, moment_order):
    mechanism_eps = 2.0 * laplace_inverse_scale
    lam = float(moment_order)
    concentration_bound = 0.5 * (mechanism_eps ** 2) * lam * (lam + 1.0)
    direct_dp_bound = mechanism_eps * lam
    return min(concentration_bound, direct_dp_bound)

def calculate_pate_privacy(vote_counts, laplace_inverse_scale=1.3, delta=1e-5, moments=32):
    """
    Compute data-dependent and data-independent privacy guarantees
    using the moments accountant.
    """
    vote_counts = np.asarray(vote_counts, dtype=np.int64)
    num_queries, classes_dim = vote_counts.shape
    orders = np.arange(1, moments + 1, dtype=np.float64)
    
    dependent_total = np.zeros(moments, dtype=np.float64)
    independent_total = np.zeros(moments, dtype=np.float64)
    q_values, vote_margins = [], []

    for counts in vote_counts:
        q = compute_q_noisy_max(counts, laplace_inverse_scale)
        q_values.append(q)
        sorted_counts = np.sort(counts)
        vote_margins.append(sorted_counts[-1] - sorted_counts[-2])
        
        for pos, order in enumerate(orders):
            dependent_total[pos] += data_dependent_log_moment(counts, laplace_inverse_scale, order)
            independent_total[pos] += data_independent_log_moment(laplace_inverse_scale, order)

    # Convert accumulated log-moments to (ε, δ)-DP
    dep_candidates = (dependent_total - math.log(delta)) / orders
    ind_candidates = (independent_total - math.log(delta)) / orders
    
    dep_idx = np.argmin(dep_candidates)
    ind_idx = np.argmin(ind_candidates)

    dep_epsilon = float(dep_candidates[dep_idx])
    ind_epsilon = float(ind_candidates[ind_idx])

    # Safety check
    if dep_epsilon > ind_epsilon + 1e-10:
        raise AssertionError(
            f"Invalid privacy result: dependent epsilon {dep_epsilon} "
            f"exceeds independent epsilon {ind_epsilon}"
        )

    return {
        "data_dependent_epsilon": dep_epsilon,
        "data_independent_epsilon": ind_epsilon,
        "q_values": np.asarray(q_values),
        "vote_margins": np.asarray(vote_margins)
    }
# =====================================================================
# COMPLIANCE UNIT TESTS
# =====================================================================

def test_dd_never_exceeds_di():
    """Ensure data-dependent ε never exceeds data-independent ε."""
    test_votes = np.array([
        [9, 0, 0],   # strong consensus
        [6, 2, 1],   # moderate consensus
        [4, 3, 2],   # weak consensus
        [3, 3, 3]    # no consensus
    ])
    result = calculate_pate_privacy(
        test_votes,
        laplace_inverse_scale=1.3,
        delta=1e-5,
        moments=32
    )
    assert (
        result["data_dependent_epsilon"]
        <= result["data_independent_epsilon"] + 1e-10
    ), result

def run_accounting_tests():
    print("\n" + "=" * 70)
    print("RUNNING PATE ACCOUNTING COMPLIANCE TESTS")
    print("=" * 70)

    # -------------------------------------------------
    # Test 1: Verify q-estimator against hand calculation
    # -------------------------------------------------
    counts = np.array([10, 6, 2], dtype=float)
    eta = 0.1
    gaps = eta * np.array([4.0, 8.0])
    expected_q = min(np.sum((gaps + 2.0) * np.exp(-gaps) / 4.0), 2.0 / 3.0)
    actual_q = compute_q_noisy_max(counts, eta)
    assert np.isclose(actual_q, expected_q), f"q discrepancy: {actual_q} vs {expected_q}"
    print("PASS: Direct multiclass q verification")

    # -------------------------------------------------
    # Test 2: Data-independent log-moment regression test
    # -------------------------------------------------
    di_test_val = data_independent_log_moment(laplace_inverse_scale=1.0, moment_order=2)
    assert np.isclose(di_test_val, 4.0), f"DI log moment discrepancy: {di_test_val} vs 4.0"
    print("PASS: Independent hand-calculated DI regression test")

    # -------------------------------------------------
    # Test 3: Data-dependent ≤ Data-independent invariant
    # -------------------------------------------------
    test_dd_never_exceeds_di()
    print("PASS: Data-Dependent <= Data-Independent invariant test (test_dd_never_exceeds_di)")
    
    print("All core compliance unit tests passed successfully!")
TestPurposeq verificationConfirms that compute_q_noisy_max matches the analytical formula for the probability that noisy-max fails.
DI regressionChecks that the data-independent log-moment for $\beta=1$, $\lambda=2$ equals the expected value $4.0$.
DD ≤ DI invariantGuarantees that the data-dependent ε is never larger than the data-independent ε across different consensus strengths.
