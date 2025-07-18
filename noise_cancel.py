#!/usr/bin/env python3
"""
RNNoise cancellation script for VaaniVerse

This script uses RNNoise to remove background noise from audio.
"""

import argparse
import json
import os
import sys
import numpy as np
import soundfile as sf
import librosa
from scipy.io import wavfile
import noisereduce as nr

def process_audio(input_file, output_file):
    """Process audio with noise reduction."""
    try:
        # Load the audio file
        y, sr = librosa.load(input_file, sr=None)
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        
        # Save the processed audio
        sf.write(output_file, reduced_noise, sr)
        
        return True
    except Exception as e:
        print(json.dumps({
            "error": f"Failed to process audio: {str(e)}"
        }))
        return False

def main():
    parser = argparse.ArgumentParser(description="Noise Cancellation")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output audio file path")
    
    args = parser.parse_args()
    
    success = process_audio(args.input, args.output)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
