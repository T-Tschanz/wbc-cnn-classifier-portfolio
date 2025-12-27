import numpy as np

def probs_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def classify_with_gray_zone(
    probs: np.ndarray,
    class_names: list[str],
    thresholds: dict | None = None,
):
    """
    probs: shape (C,) softmax probabilities for one image
    class_names: ordered list of class labels
    thresholds: {
        "global": {"p1": 0.85, "margin": 0.15, "entropy": 1.20},
        "per_class": { "neutrophil":{"p1":0.88,"margin":0.18,"entropy":1.10}, ... }
    }
    Returns: dict with fields:
      top1_idx, top1_label, top1_prob, top2_label, top2_prob,
      entropy, is_confident, decision ("CONFIDENT"|"GRAY_ZONE")
    """
    if thresholds is None:
        thresholds = {"global": {"p1": 0.85, "margin": 0.15, "entropy": 1.20}, "per_class": {}}

    order = np.argsort(probs)[::-1]
    top1, top2 = order[0], order[1]
    p1, p2 = float(probs[top1]), float(probs[top2])
    margin = p1 - p2
    ent = probs_entropy(probs)

    # resolve thresholds (per-class overrides global)
    label = class_names[top1]
    t_pc = thresholds.get("per_class", {}).get(label, {})
    p1_thr = t_pc.get("p1", thresholds["global"]["p1"])
    margin_thr = t_pc.get("margin", thresholds["global"]["margin"])
    ent_thr = t_pc.get("entropy", thresholds["global"]["entropy"])

    is_confident = (p1 >= p1_thr) and (margin >= margin_thr) and (ent <= ent_thr)
    decision = "CONFIDENT" if is_confident else "GRAY_ZONE"
    return {
        "top1_idx": int(top1),
        "top1_label": label,
        "top1_prob": p1,
        "top2_label": class_names[top2],
        "top2_prob": p2,
        "entropy": ent,
        "margin": margin,
        "is_confident": is_confident,
        "decision": decision,
    }
