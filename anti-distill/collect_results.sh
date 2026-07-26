#!/usr/bin/env bash
set -euo pipefail

#-----------------------------------------------------
# Script: copy toàn bộ file .yaml/.yml từ thư mục hiện
# tại (và các thư mục con) sang thư mục đích, giữ nguyên
# cấu trúc đường dẫn tương đối.
#
# Cách dùng:
#   ./copy_yaml.sh <thu_muc_dich>
#
# Ví dụ:
#   ./copy_yaml.sh ./backup_yaml
#-----------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Cách dùng: $0 <thu_muc_dich>" >&2
    exit 1
fi

DEST_DIR="${1:-results_yaml}"
SRC_DIR="."

mkdir -p "$DEST_DIR"

count=0
while IFS= read -r -d '' file; do
    # Đường dẫn tương đối so với thư mục hiện tại
    rel_path="${file#./}"
    dest_path="$DEST_DIR/$rel_path"

    # Tạo thư mục con tương ứng nếu chưa có
    mkdir -p "$(dirname "$dest_path")"

    cp -p "$file" "$dest_path"
    echo "[OK] $rel_path"
    count=$((count + 1))
done < <(find "$SRC_DIR" -type f \( -iname "*.yaml" -o -iname "*.yml" \) -print0)

echo ""
echo "Đã copy $count file .yaml/.yml sang '$DEST_DIR', giữ nguyên cấu trúc thư mục."