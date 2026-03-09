#!/bin/bash
# 青空文庫テキストを取得・クリーニングするスクリプト
# aozorahack/aozorabunko_text + aozorabunko-extractor を使用
#
# 前提: pip install aozorabunko-extractor
#
# Usage:
#   ./scripts/fetch_aozora.sh <card_no/file_dir> <output_file>
#
# Examples:
#   ./scripts/fetch_aozora.sh 000879/92_ruby_164     data/kumo_no_ito.txt       # 蜘蛛の糸 (芥川)
#   ./scripts/fetch_aozora.sh 000879/127_ruby_150    data/rashomon.txt           # 羅生門 (芥川)
#   ./scripts/fetch_aozora.sh 000879/42_ruby_154     data/hana.txt              # 鼻 (芥川)
#   ./scripts/fetch_aozora.sh 000035/1567_ruby_4948  data/hashire_merosu.txt    # 走れメロス (太宰)
#   ./scripts/fetch_aozora.sh 000081/43754_ruby_17058 data/chuumon.txt          # 注文の多い料理店 (宮沢)

set -euo pipefail

REPO_DIR="/tmp/aozorabunko_text"
REPO_URL="https://github.com/aozorahack/aozorabunko_text.git"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <card_no/file_dir> <output_file>" >&2
    exit 1
fi

CARD_PATH="$1"
OUTPUT="$2"
CARD_NO=$(dirname "$CARD_PATH")
FILE_DIR=$(basename "$CARD_PATH")

# Clone repo if not present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning aozorabunko_text..." >&2
    git clone --depth 1 "$REPO_URL" "$REPO_DIR" 2>&1 | tail -1 >&2
fi

INPUT_DIR="${REPO_DIR}/cards/${CARD_NO}/files/${FILE_DIR}"
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: $INPUT_DIR not found" >&2
    exit 1
fi

# Extract and clean using aozorabunko-extractor
TMPDIR=$(mktemp -d)
python3 -c "from aozorabunko_extractor import cli; cli.main()" \
    -i "$INPUT_DIR" -o "$TMPDIR" 2>/dev/null

cp "$TMPDIR/all.txt" "$OUTPUT"
rm -rf "$TMPDIR"

CHARS=$(wc -m < "$OUTPUT")
echo "Saved to $OUTPUT (${CHARS} chars)" >&2
