#!/bin/bash
# 青空文庫からテキストを取得するスクリプト
# Usage: ./scripts/fetch_aozora.sh <URL> <output_file>
# Example: ./scripts/fetch_aozora.sh https://www.aozora.gr.jp/cards/000879/files/92_14545.html data/kumo_no_ito.txt

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <aozora_html_url> <output_file>" >&2
    echo "Example: $0 https://www.aozora.gr.jp/cards/000879/files/92_14545.html data/kumo_no_ito.txt" >&2
    exit 1
fi

URL="$1"
OUTPUT="$2"

curl -s "$URL" \
  | iconv -f SHIFT_JIS -t UTF-8 2>/dev/null \
  | python3 -c "
import sys, re, html

text = sys.stdin.read()

# Remove ruby readings: keep base text only
text = re.sub(r'<ruby><rb>(.*?)</rb><rp>[（(]</rp><rt>.*?</rt><rp>[）)]</rp></ruby>', r'\1', text)
text = re.sub(r'<ruby>(.*?)<rt>.*?</rt></ruby>', r'\1', text)

# Remove all HTML tags
text = re.sub(r'<[^>]+>', '\n', text)

# Decode HTML entities
text = html.unescape(text)

# Remove Aozora annotations
text = re.sub(r'[\[［]＃[^\]］]*[\]］]', '', text)

# Clean whitespace
text = re.sub(r'[ \t]+', '', text)
text = re.sub(r'\n{2,}', '\n', text)
text = text.strip()

lines = text.split('\n')

# Find content boundaries
start = None
end = len(lines)
for i, line in enumerate(lines):
    stripped = line.strip()
    # First section marker (一, 上, etc.) or first indented paragraph
    if start is None and (stripped in ['一', '上'] or stripped.startswith('　')):
        start = i
    # Date at the end like （大正...） or （昭和...）
    if re.match(r'[（(](大正|明治|昭和|平成|令和)', stripped):
        end = i + 1

if start is None:
    start = 0

print('\n'.join(lines[start:end]).strip())
" > "$OUTPUT"

CHARS=$(wc -m < "$OUTPUT")
UNIQUE=$(grep -o . "$OUTPUT" | sort -u | wc -l)
echo "Saved to $OUTPUT"
echo "Total chars: $CHARS"
echo "Unique chars: $UNIQUE"
