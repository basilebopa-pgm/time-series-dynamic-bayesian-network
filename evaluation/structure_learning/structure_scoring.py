from sklearn.metrics import precision_score, recall_score, f1_score

def normalize_edge(source, target):
    source_name, source_slice = source[0], source[1]
    target_name, target_slice = target[0], target[1]
    lag = target_slice - source_slice

    if lag == 0:
        return ("intra", source_name, target_name)

    if lag > 0:
        return ("inter", source_name, target_name, lag)

    raise ValueError(f"Unexpected DBN edge: {source} -> {target}")


def extract_edges(model):
    edges = set()

    for source, target in model.edges():
        edges.add(normalize_edge(source, target))

    return edges


def evaluate_edges(ground_truth_edges, learned_edges):
    true_positive = len(ground_truth_edges & learned_edges)
    false_positive = len(learned_edges - ground_truth_edges)
    false_negative = len(ground_truth_edges - learned_edges)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    shd = false_positive + false_negative

    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "SHD": shd
    }

def evaluate_structure(ground_truth, learned_model, model_name):
    true_edges = extract_edges(ground_truth)
    learned_edges = extract_edges(learned_model)

    print(f"\n{'=' * 50}")
    print(f"Structure evaluation: {model_name}")
    print(f"{'=' * 50}")

    categories = {
        "Intra-slice": "intra",
        "Inter-slice": "inter",
        "Combined": None
    }

    results = {}

    for label, category in categories.items():
        if category is None:
            reference = true_edges
            prediction = learned_edges
        else:
            reference = {e for e in true_edges if e[0] == category}
            prediction = {e for e in learned_edges if e[0] == category}

        metrics = evaluate_edges(reference, prediction)
        results[label] = metrics

        print(f"\n{label}")
        print(f"Ground-truth edges: {len(reference)}")
        print(f"Learned edges:      {len(prediction)}")

        for metric, value in metrics.items():
            if metric in ("Precision", "Recall", "F1"):
                print(f"{metric:12}: {value:.4f}")
            else:
                print(f"{metric:12}: {value}")

    print("\nMissing edges:", sorted(true_edges - learned_edges))
    print("Extra edges:  ", sorted(learned_edges - true_edges))

    return results

