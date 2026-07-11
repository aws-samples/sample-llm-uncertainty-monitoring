# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Uncertainty explainability via word-cloud visualization.

Consumes batch evaluation records (from the batch JSON written by
``run_batch_evaluation.py``) and renders a word cloud over the question
text of the *uncertain tail* — records whose ``set_size >= min_set_size``.

Frequent terms in the uncertain tail surface the topics, vocabulary, and
entities the model struggles with, complementing the quantitative
monitoring signal (mean set size) with a qualitative "what is the model
uncertain about?" view.

The pipeline is fully offline and on-demand::

    S3 batch JSON  →  explain_uncertainty(records)  →  word cloud PNG (local + S3)
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import boto3
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

__all__ = ["explain_uncertainty", "compare_certain_uncertain"]

# Extended stopwords for MMLU-style academic questions — common function words
# and roman numerals that appear frequently but carry no topical signal.
_EXTENDED_STOPWORDS: set[str] = set(STOPWORDS) | {
    "will",
    "used",
    "using",
    "use",
    "one",
    "two",
    "three",
    "four",
    "following",
    "would",
    "may",
    "also",
    "could",
    "must",
    "many",
    "often",
    "well",
    "however",
    "called",
    "known",
    "refers",
    "referred",
    "among",
    "within",
    "iii",
    "ii",
    "iv",
    "based",
    "given",
    "include",
    "includes",
    "likely",
    "true",
    "false",
    "best",
    "least",
    "example",
    "examples",
    "result",
    "results",
    "term",
    "terms",
    "type",
    "types",
    "form",
    "forms",
    "can",
    "new",
    "first",
    "make",
    "way",
    "made",
    "see",
    "say",
}


def explain_uncertainty(
    records: list[dict[str, Any]],
    min_set_size: int = 2,
    output_local_path: str = "wordcloud.png",
    output_s3_uri: str | None = None,
    extra_stopwords: set[str] | None = None,
) -> str:
    """Render a word cloud over the uncertain tail of batch records.

    Args:
        records: list of record dicts from a batch JSON, each containing
            at least ``set_size`` and ``question``.
        min_set_size: minimum prediction set size to include a record in
            the uncertain tail (default 2).
        output_local_path: local file path for the rendered PNG
            (default ``"wordcloud.png"``).
        output_s3_uri: optional S3 URI to upload the PNG after rendering.
        extra_stopwords: additional stopwords to exclude beyond the
            default ``wordcloud.STOPWORDS`` set.

    Returns:
        The local path where the PNG was saved.

    Raises:
        ValueError: if fewer than 3 records meet the ``min_set_size``
            threshold (not enough data for a meaningful word cloud).
    """
    # Filter to the uncertain tail
    selected = [r for r in records if r.get("set_size", 0) >= min_set_size]
    if len(selected) < 3:
        raise ValueError(
            f"only {len(selected)} records meet set_size >= {min_set_size}; "
            f"need at least 3 for a meaningful word cloud"
        )

    # Build stopword set
    stopwords = set(_EXTENDED_STOPWORDS)
    if extra_stopwords:
        stopwords |= extra_stopwords

    # Tokenize and aggregate term frequencies
    freqs: dict[str, int] = {}
    for record in selected:
        question = record.get("question", "")
        tokens = re.findall(r"[a-z]+", question.lower())
        for token in tokens:
            if token not in stopwords and len(token) > 1:
                freqs[token] = freqs.get(token, 0) + 1

    if not freqs:
        raise ValueError("no tokens remain after stopword removal")

    # Render word cloud
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(freqs)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_local_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Upload to S3 if requested
    if output_s3_uri:
        parsed = urlparse(output_s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        boto3.client("s3").upload_file(output_local_path, bucket, key)

    return output_local_path


def compare_certain_uncertain(
    records: list[dict[str, Any]],
    top_n: int = 10,
    output_local_path: str = "wordcloud_comparison.png",
    output_s3_uri: str | None = None,
    extra_stopwords: set[str] | None = None,
) -> dict[str, Any]:
    """Compare word frequencies between certain and uncertain predictions.

    Generates side-by-side word clouds and a top-N terms comparison table.

    Args:
        records: list of record dicts from a batch JSON, each containing
            at least ``set_size`` and ``question``.
        top_n: number of top terms to return per group (default 10).
        output_local_path: local file path for the comparison PNG.
        output_s3_uri: optional S3 URI to upload the PNG.
        extra_stopwords: additional stopwords to exclude.

    Returns:
        A dict with keys ``certain_top_terms``, ``uncertain_top_terms``
        (each a list of (term, count) tuples), and ``output_path``.

    Raises:
        ValueError: if either group has fewer than 3 records.
    """
    certain = [r for r in records if r.get("set_size", 0) == 1]
    uncertain = [r for r in records if r.get("set_size", 0) >= 2]

    if len(certain) < 3:
        raise ValueError(f"only {len(certain)} certain records; need at least 3")
    if len(uncertain) < 3:
        raise ValueError(f"only {len(uncertain)} uncertain records; need at least 3")

    stopwords = set(_EXTENDED_STOPWORDS)
    if extra_stopwords:
        stopwords |= extra_stopwords

    def _get_freqs(group: list[dict[str, Any]]) -> dict[str, int]:
        freqs: dict[str, int] = {}
        for record in group:
            tokens = re.findall(r"[a-z]+", record.get("question", "").lower())
            for token in tokens:
                if token not in stopwords and len(token) > 1:
                    freqs[token] = freqs.get(token, 0) + 1
        return freqs

    certain_freqs = _get_freqs(certain)
    uncertain_freqs = _get_freqs(uncertain)

    if not certain_freqs or not uncertain_freqs:
        raise ValueError("no tokens remain after stopword removal in one or both groups")

    # Differential terms: words disproportionately associated with uncertainty
    # ratio = uncertain_count / (uncertain_count + certain_count)
    # Only consider words appearing > min_total_count times across both groups
    min_total_count = 5
    all_words = set(certain_freqs.keys()) | set(uncertain_freqs.keys())
    uncertain_differential: dict[str, float] = {}
    certain_differential: dict[str, float] = {}
    for word in all_words:
        if len(word) < 3:
            continue
        uc = uncertain_freqs.get(word, 0)
        cc = certain_freqs.get(word, 0)
        total = uc + cc
        if total <= min_total_count:
            continue
        ratio = uc / total
        uncertain_differential[word] = ratio
        certain_differential[word] = 1.0 - ratio

    if not uncertain_differential or not certain_differential:
        raise ValueError("no words meet the min_total_count threshold for differential analysis")

    # Generate side-by-side word clouds using differential ratios
    wc_certain = WordCloud(
        width=600, height=400, background_color="white", colormap="Blues"
    ).generate_from_frequencies(certain_differential)
    wc_uncertain = WordCloud(
        width=600, height=400, background_color="white", colormap="Reds"
    ).generate_from_frequencies(uncertain_differential)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(wc_certain, interpolation="bilinear")
    axes[0].set_title(f"Certain (set_size=1, n={len(certain)})", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(wc_uncertain, interpolation="bilinear")
    axes[1].set_title(f"Uncertain (set_size≥2, n={len(uncertain)})", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_local_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Top-N by differential ratio
    top_uncertain_diff = sorted(uncertain_differential.items(), key=lambda x: -x[1])[:top_n]
    top_certain_diff = sorted(certain_differential.items(), key=lambda x: -x[1])[:top_n]

    # Upload to S3 if requested
    if output_s3_uri:
        parsed = urlparse(output_s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        boto3.client("s3").upload_file(output_local_path, bucket, key)

    return {
        "top_uncertain_differential": top_uncertain_diff,
        "top_certain_differential": top_certain_diff,
        "output_path": output_local_path,
        "n_certain": len(certain),
        "n_uncertain": len(uncertain),
    }
