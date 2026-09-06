import json
import os

from image_loader import load_image
from quality_check import quality_report
from alignment import align_images
from change_detection import compute_change_mask, clean_mask, change_summary
from evidence import build_evidence_image
from confidence import evidence_confidence


def run_pipeline(before_path, after_path, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    before = load_image(before_path)
    after = load_image(after_path)

    q_before = quality_report(before)
    q_after = quality_report(after)

    aligned_after, valid_mask, align_diag = align_images(before, after)

    raw_mask, _signals = compute_change_mask(before, aligned_after, valid_mask=valid_mask)
    final_mask, regions = clean_mask(raw_mask)
    summary = change_summary(final_mask, regions)

    conf_score, conf_label = evidence_confidence(
        align_diag["alignment_confidence"],
        q_before["usable"], q_after["usable"],
        summary["significant_regions"],
    )

    evidence_path = build_evidence_image(before, aligned_after, final_mask, f"{out_dir}/evidence_report.jpg", regions=regions)

    result = {
        "quality": {"before": q_before, "after": q_after},
        "alignment": align_diag,
        "change": summary,
        "evidence_confidence": {"score": conf_score, "label": conf_label},
        "evidence_image": evidence_path,
    }

    with open(f"{out_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = run_pipeline("data/before/before.jpeg", "data/after/after.jpeg")
    print(json.dumps(result, indent=2))
