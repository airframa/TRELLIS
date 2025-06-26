#!/bin/bash
# Optimized SharePoint Direct Server Download
# Uses cookies for authentication and parallel downloads for speed

set -e  # Exit on error

# Configuration
DOWNLOAD_DIR="/data/francMB/threedrealcar/pipeline-workspace/downloads"
COOKIE_FILE="$HOME/sharepoint_cookies.txt"
LOG_FILE="$HOME/sharepoint_download.log"
PARALLEL_DOWNLOADS=3  # Adjust based on server capacity

# Ensure download directory exists
mkdir -p "$DOWNLOAD_DIR"

# Download function with progress tracking
download_file() {
    local filename=$1
    local url=$2
    local output_path="$DOWNLOAD_DIR/$filename"
    
    # Skip if already downloaded
    if [ -f "$output_path" ]; then
        local size=$(stat -c%s "$output_path" 2>/dev/null || stat -f%z "$output_path" 2>/dev/null)
        if [ "$size" -gt 1000000 ]; then  # > 1MB
            echo "[$(date)] ✓ $filename already exists ($(numfmt --to=iec-i --suffix=B $size))" | tee -a "$LOG_FILE"
            return 0
        fi
    fi
    
    echo "[$(date)] ⬇️  Downloading $filename..." | tee -a "$LOG_FILE"
    
    # Use curl with optimizations
    curl -L \
        --cookie "$COOKIE_FILE" \
        --cookie-jar "$COOKIE_FILE" \
        --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' \
        --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
        --header 'Accept-Language: en-US,en;q=0.5' \
        --header 'Connection: keep-alive' \
        --header 'Upgrade-Insecure-Requests: 1' \
        --compressed \
        --progress-bar \
        --continue-at - \
        --retry 5 \
        --retry-delay 10 \
        --retry-max-time 0 \
        --max-time 0 \
        --output "$output_path" \
        "$url" 2>&1 | tee -a "$LOG_FILE"
    
    # Verify download
    if [ -f "$output_path" ]; then
        local size=$(stat -c%s "$output_path" 2>/dev/null || stat -f%z "$output_path" 2>/dev/null)
        echo "[$(date)] ✅ Completed $filename ($(numfmt --to=iec-i --suffix=B $size))" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$(date)] ❌ Failed to download $filename" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Export function for parallel execution
export -f download_file
export DOWNLOAD_DIR COOKIE_FILE LOG_FILE

# List of downloads
declare -a DOWNLOADS=(
    "0-200.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=8672d3ab%2D26e1%2D4824%2D8c26%2Dfab0be792e42"
    "200-400.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=b2d021e0%2Debbc%2D41e4%2D81e7%2D160d1ed44f20"
    "400-600.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=b0378ad5%2Dbd93%2D495a%2Daa9a%2Db5e47026be98"
    "600-800.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=6b75fc91%2D81c5%2D48df%2D9803%2D2d9af45b9932"
    "800-1000.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=d290c6a5%2D8350%2D4396%2Db0fb%2D1d6667cdfb4d"
    "1000-1200.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=fc6f3946%2Da344%2D489b%2Dab54%2D84d60c07c7d5"
    "1200-1400.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=9178ffe1%2D725e%2D40e9%2D9cf2%2D4e2a988f6217"
    "1400-1600.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=4c8a0091%2D55b3%2D45e1%2Da026%2Dcdf200b749e6"
    "1600-1800.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=43f56014%2D2b63%2D48a8%2D85f1%2D585585a7812a"
    "1800-2045.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=89428d69%2Ddfcf%2D47e3%2Dbbcc%2D0b32cb7c9084"
    "HQ200.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=9028644f%2Db2ae%2D4777%2Dbd27%2D0a7bc40c7982"
    "HQ300.zip|https://studentutsedu-my.sharepoint.com/personal/xiaobiao_du_student_uts_edu_au/_layouts/15/download.aspx?UniqueId=fe634a25%2D5df8%2D406d%2Dbd27%2D0a7bc40c7982"
)

echo "=== SharePoint Direct Server Download ===" | tee "$LOG_FILE"
echo "Download directory: $DOWNLOAD_DIR" | tee -a "$LOG_FILE"
echo "Cookie file: $COOKIE_FILE" | tee -a "$LOG_FILE"
echo "Parallel downloads: $PARALLEL_DOWNLOADS" | tee -a "$LOG_FILE"
echo "Total files: ${#DOWNLOADS[@]}" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# Check for cookie file
if [ ! -f "$COOKIE_FILE" ]; then
    echo "❌ Cookie file not found!" | tee -a "$LOG_FILE"
    echo | tee -a "$LOG_FILE"
    echo "Please follow these steps:" | tee -a "$LOG_FILE"
    echo "1. Install 'cookies.txt' extension in Chrome/Firefox" | tee -a "$LOG_FILE"
    echo "2. Login to SharePoint/OneDrive" | tee -a "$LOG_FILE"
    echo "3. Export cookies for *.sharepoint.com" | tee -a "$LOG_FILE"
    echo "4. Transfer to server: scp cookies.txt $USER@$(hostname):~/sharepoint_cookies.txt" | tee -a "$LOG_FILE"
    exit 1
fi

# Function to show progress
show_progress() {
    while true; do
        sleep 30
        echo
        echo "=== Download Progress ==="
        for file in "$DOWNLOAD_DIR"/*.zip; do
            if [ -f "$file" ]; then
                size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
                printf "%-20s %s\n" "$(basename "$file")" "$(numfmt --to=iec-i --suffix=B $size)"
            fi
        done
        echo
    done
}

# Start progress monitor in background
show_progress &
PROGRESS_PID=$!

# Cleanup function
cleanup() {
    echo "[$(date)] Cleaning up..." | tee -a "$LOG_FILE"
    kill $PROGRESS_PID 2>/dev/null || true
    exit
}
trap cleanup EXIT INT TERM

# Parallel download using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    echo "[$(date)] Using GNU parallel for downloads" | tee -a "$LOG_FILE"
    printf '%s\n' "${DOWNLOADS[@]}" | parallel -j $PARALLEL_DOWNLOADS --colsep '|' download_file {1} {2}
else
    echo "[$(date)] Using xargs for parallel downloads" | tee -a "$LOG_FILE"
    printf '%s\n' "${DOWNLOADS[@]}" | xargs -P $PARALLEL_DOWNLOADS -I {} bash -c 'IFS="|" read -r filename url <<< "$1"; download_file "$filename" "$url"' -- {}
fi

# Final summary
echo | tee -a "$LOG_FILE"
echo "=== Download Summary ===" | tee -a "$LOG_FILE"
successful=0
failed=0
total_size=0

for item in "${DOWNLOADS[@]}"; do
    IFS='|' read -r filename url <<< "$item"
    filepath="$DOWNLOAD_DIR/$filename"
    
    if [ -f "$filepath" ]; then
        size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
        if [ "$size" -gt 1000000 ]; then
            echo "✅ $filename - $(numfmt --to=iec-i --suffix=B $size)" | tee -a "$LOG_FILE"
            ((successful++))
            ((total_size+=size))
        else
            echo "❌ $filename - Invalid size" | tee -a "$LOG_FILE"
            ((failed++))
        fi
    else
        echo "❌ $filename - Not found" | tee -a "$LOG_FILE"
        ((failed++))
    fi
done

echo | tee -a "$LOG_FILE"
echo "Total successful: $successful/${#DOWNLOADS[@]}" | tee -a "$LOG_FILE"
echo "Total size: $(numfmt --to=iec-i --suffix=B $total_size)" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

if [ "$successful" -eq "${#DOWNLOADS[@]}" ]; then
    echo "✅ All downloads completed successfully!" | tee -a "$LOG_FILE"
    echo "You can now run the processing pipeline." | tee -a "$LOG_FILE"
else
    echo "⚠️  Some downloads failed. Check the log file: $LOG_FILE" | tee -a "$LOG_FILE"
fi