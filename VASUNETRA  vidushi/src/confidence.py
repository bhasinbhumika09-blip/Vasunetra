def evidence_confidence(alignment_conf, quality_ok_before, quality_ok_after, num_regions):
    """
    Combines alignment reliability + image quality + presence of coherent
    change regions into a single evidence confidence score.
    This is an EVIDENCE confidence score, not a hazard probability.
    """
    score = alignment_conf
    if not quality_ok_before or not quality_ok_after:
        score *= 0.6
    if num_regions == 0:
        score *= 0.7

    if score >= 80:
        label = "HIGH"
    elif score >= 55:
        label = "MEDIUM"
    else:
        label = "LOW"
    return round(score, 1), label
