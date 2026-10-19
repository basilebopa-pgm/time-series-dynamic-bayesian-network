import numpy as np
import torch

def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)

    if np.any(sigma_p <= 0) or np.any(sigma_q <= 0):
        raise ValueError("Standard deviations must be positive.")

    return (np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5)

def estimate_kl(log_prob_gt, log_prob_learned):

    log_prob_gt = np.asarray(log_prob_gt)
    log_prob_learned = np.asarray(log_prob_learned)

    if log_prob_gt.shape != log_prob_learned.shape:
        raise ValueError("Both arrays must have the same shape.")

    if not np.all(np.isfinite(log_prob_gt)):
        raise ValueError("Ground-truth log probabilities must be finite.")

    if not np.all(np.isfinite(log_prob_learned)):
        raise ValueError("Learned-model log probabilities must be finite.")

    return np.mean(log_prob_gt - log_prob_learned)

def mean_conditional_kl(true_means, true_variances, learned_means, learned_variances):
    true_means = np.asarray(true_means, dtype=float)
    learned_means = np.asarray(learned_means, dtype=float)

    true_variances = np.asarray(true_variances, dtype=float)
    learned_variances = np.asarray(learned_variances, dtype=float)

    true_sigmas = np.sqrt(np.maximum(true_variances, 1e-8))
    learned_sigmas = np.sqrt(np.maximum(learned_variances, 1e-8))

    kl_values = gaussian_kl(true_means, true_sigmas,learned_means, learned_sigmas,)

    return float(np.mean(kl_values))

def get_linear_distribution(cpd, parent_values):
    
    beta = np.asarray(cpd.beta).reshape(-1)
    values = np.asarray([parent_values[parent] for parent in cpd.evidence], dtype=float)
    mean = beta[0] + np.dot(beta[1:], values)
    variance = float(cpd.std) ** 2

    return mean, variance

def get_neural_distribution(cpd, parent_values):
    
    cpd.eval()
    parents = torch.as_tensor(parent_values,dtype=next(cpd.parameters()).dtype)

    with torch.no_grad():
        mean, variance = cpd(parents)

    return mean.cpu().numpy(), variance.cpu().numpy()



def evaluate_parameters(ground_truth, learned_model, test_data):

    gt_cpds = {cpd.variable: cpd for cpd in ground_truth.get_cpds()}
    learned_cpds = {cpd.variable: cpd for cpd in learned_model.get_cpds()}

    per_variable_kl = {}
    per_variable_rmse = {}

    shared_variables = (set(gt_cpds) & set(learned_cpds) & set(test_data.columns))

    for variable in sorted(shared_variables):
        gt_cpd = gt_cpds[variable]
        learned_cpd = learned_cpds[variable]

        gt_parents = list(getattr(gt_cpd, "evidence", []) or [])
        learned_parents = list(getattr(learned_cpd, "evidence", []) or [])

        if set(gt_parents) != set(learned_parents):
            print(f"Skipping {variable}: parent sets differ.")
            continue

        if any(parent not in test_data.columns for parent in gt_parents):
            print(f"Skipping {variable}: missing parent column.")
            continue

        if not hasattr(gt_cpd, "beta") or not hasattr(gt_cpd, "std"):
            raise TypeError(f"Ground-truth CPD for {variable} must expose beta and std.")

        X = test_data[gt_parents].to_numpy(dtype=float)
        beta = np.asarray(gt_cpd.beta, dtype=float).reshape(-1)

        if len(beta) != len(gt_parents) + 1:
            raise ValueError(f"Unexpected beta shape for {variable}.")

        true_means = beta[0] + ( X @ beta[1:] if gt_parents else 0.0)
        true_variances = np.full(len(test_data), float(gt_cpd.std) ** 2)

        if hasattr(learned_cpd, "beta") and hasattr(learned_cpd, "std"):
            learned_beta = np.asarray(learned_cpd.beta, dtype=float).reshape(-1)

            X_learned = test_data[learned_parents].to_numpy(dtype=float)

            learned_means = learned_beta[0] + (X_learned @ learned_beta[1:]if learned_parents else 0.0)
            learned_variances = np.full(len(test_data), float(learned_cpd.std) ** 2)

        else:
            parent_values = test_data[learned_parents].to_numpy(dtype=np.float32)
            learned_means, learned_variances = get_neural_distribution( learned_cpd, parent_values)
            learned_means = np.asarray(learned_means).reshape(-1)
            learned_variances = np.asarray(learned_variances).reshape(-1)

        per_variable_kl[variable] = mean_conditional_kl(true_means,true_variances,learned_means,learned_variances,)

    variable_name = getattr(variable, "variable", variable)

    if variable_name in test_data.columns:
        observed = test_data[variable_name].to_numpy(dtype=float)

        per_variable_rmse[variable] = float(
            np.sqrt(np.mean((observed - learned_means) ** 2))
        )

    if not per_variable_kl:
        raise ValueError(
            "No compatible CPDs were evaluated. Check node names, "
            "parent sets, and test-data columns."
        )

    print("Test-data columns:")
    print(test_data.columns.tolist())

    print("\nGround-truth CPD variables:")
    print([cpd.variable for cpd in ground_truth.get_cpds()])

    print("\nModel 1 CPD variables:")
    print([cpd.variable for cpd in learned_model.get_cpds()])
    return {
        "Overall": {
            "Mean Conditional KL": float(np.mean(list(per_variable_kl.values()))),
            "RMSE": float(np.mean(list(per_variable_rmse.values()))),
        },
        "Per-variable KL": per_variable_kl,
    }
