#!/bin/bash
DATASET="$HOME/.cache/huggingface/lerobot/local/go2_suction_extract_replay_v1_head3"

convert_one() {
    src="$1"
    codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$src" 2>/dev/null)
    if [ "$codec" = "h264" ]; then
        echo "SKIP: $src"
        return
    fi
    tmp="${src}.tmp.mp4"
    ffmpeg -loglevel error -y -i "$src" -c:v libx264 -profile:v high -crf 18 -preset fast -pix_fmt yuv420p "$tmp"
    if [ -f "$tmp" ]; then
        mv "$tmp" "$src"
        echo "OK: $(basename $src)"
    else
        echo "FAIL: $src"
    fi
}

export -f convert_one

echo "Converting HEVC -> H.264 using 8 parallel workers..."
find "$DATASET/videos" -name "*.mp4" | parallel -j8 convert_one {}

echo "Updating meta/info.json..."
sed -i 's/"video.codec": "hevc"/"video.codec": "h264"/g' "$DATASET/meta/info.json"

echo "Done!"
find "$DATASET/videos" -name "*.mp4" | xargs -I{} ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 {} 2>/dev/null | sort | uniq -c
