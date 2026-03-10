#!/usr/bin/env python3
"""Fetch Japanese Wikipedia articles and save as plain text files.

Usage:
    pip install datasets
    python scripts/fetch_wikipedia.py [--chars 1000000] [--output data]

Fetches articles from HuggingFace wikimedia/wikipedia dataset,
filters by length, and saves to the output directory.
"""

import argparse
import os
import re


def clean_text(text: str) -> str:
    """Remove Wikipedia markup artifacts and clean whitespace."""
    # Remove section headers (== Title ==)
    text = re.sub(r"={2,}\s*.+?\s*={2,}", "", text)
    # Remove reference tags and templates
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Fetch Japanese Wikipedia articles")
    parser.add_argument("--chars", type=int, default=1_000_000,
                        help="Target total character count (default: 1000000)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--min-chars", type=int, default=2000,
                        help="Minimum article length in chars (default: 2000)")
    parser.add_argument("--max-chars", type=int, default=30000,
                        help="Maximum chars per article to save (default: 30000)")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading Japanese Wikipedia from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)

    os.makedirs(args.output, exist_ok=True)

    total_chars = 0
    n_articles = 0

    for article in ds:
        text = clean_text(article["text"])

        # Skip short articles
        if len(text) < args.min_chars:
            continue

        # Truncate long articles
        if len(text) > args.max_chars:
            # Cut at sentence boundary
            cut = text[:args.max_chars]
            last_period = cut.rfind("。")
            if last_period > args.max_chars // 2:
                text = cut[:last_period + 1]
            else:
                text = cut

        title = article["title"]
        # Sanitize filename
        safe_title = re.sub(r'[/\\:*?"<>|]', "_", title)
        filename = f"wiki_{n_articles:04d}_{safe_title}.txt"
        filepath = os.path.join(args.output, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        n_articles += 1
        total_chars += len(text)

        if n_articles % 10 == 0:
            print(f"  {n_articles} articles, {total_chars:,} chars...", flush=True)

        if total_chars >= args.chars:
            break

    print(f"\nDone: {n_articles} articles, {total_chars:,} chars in {args.output}/")


if __name__ == "__main__":
    main()
