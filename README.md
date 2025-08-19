datatape: Audio Data Tape Utility
=================================

This tool encodes and decodes files to and from audio WAV format, suitable for storage or transmission on analog media like cassette tape.

Features:
---------
- Encodes any file as a WAV audio signal using multiple frequency bands.
- Decodes WAV files back to the original data, with error detection.
- Butterworth low-pass filter for noise reduction (tape-friendly).
- Simple block parity error correction.

Usage:
------

Encoding:
    ./datatape -encode -input <file> -output <audio.wav>

Decoding:
    ./datatape -decode -input <audio.wav> -output <recovered.txt>

Options:
    -sr       Sample rate (default: 44100)
    -sps      Samples per symbol (default: 2205)
    -bands    Number of frequency bands (default: 4)
    -freqs    Comma-separated custom frequencies
    -help     Show help

Example:
    ./datatape -encode -input input.txt -output output.wav
    ./datatape -decode -input output.wav -output recovered.txt

Notes:
------
- Parity error detection warns about possible corruption.
- For best results, use high-quality tape and playback equipment.

License:
--------
MIT License

Author:
-------
Mago Arsaev 2025
