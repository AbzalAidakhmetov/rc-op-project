#!/bin/bash
#
# Download VCTK Corpus for Voice Conversion Training
# ~110 English speakers with various accents
#
# Dataset size: ~11GB (wav48_silence_trimmed)
# After download: data/vctk/wav48_silence_trimmed/p{ID}/*.flac
#

set -e

DATA_DIR="data/vctk"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================"
echo "Downloading VCTK Corpus 0.92"
echo "========================================"
echo "This will download ~11GB of data."
echo ""

# Download from official source (University of Edinburgh)
URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

if [ -f "VCTK-Corpus-0.92.zip" ]; then
    echo "Archive already exists, skipping download."
else
    echo "Downloading VCTK-Corpus-0.92.zip..."
    wget --no-check-certificate -c "$URL" -O VCTK-Corpus-0.92.zip
fi

if [ -d "wav48_silence_trimmed" ]; then
    echo "Already extracted, skipping."
else
    echo "Extracting..."
    unzip -q VCTK-Corpus-0.92.zip
    
    # Move files to cleaner structure
    if [ -d "VCTK-Corpus-0.92" ]; then
        mv VCTK-Corpus-0.92/wav48_silence_trimmed .
        mv VCTK-Corpus-0.92/txt .
        rm -rf VCTK-Corpus-0.92
    fi
fi

# Count speakers and files
NUM_SPEAKERS=$(ls -d wav48_silence_trimmed/p* 2>/dev/null | wc -l)
NUM_FILES=$(find wav48_silence_trimmed -name "*.flac" 2>/dev/null | wc -l)

echo ""
echo "========================================"
echo "VCTK Download Complete!"
echo "========================================"
echo "  Location: $DATA_DIR/wav48_silence_trimmed/"
echo "  Speakers: $NUM_SPEAKERS"
echo "  Files: $NUM_FILES"
echo ""
echo "To train: python main.py --dataset vctk"
