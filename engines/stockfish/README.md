# Stockfish

Place the Stockfish executable used for the experiments in this directory.

The Stockfish binary is not included in this repository because it exceeds GitHub's file size limit.

## Version Used

The experiments in this project use:

- Stockfish 18
- Windows x86-64 AVX2 build

The executable used during development was:

`stockfish-windows-x86-64-avx2.exe`

## Setup

Download Stockfish from the official Stockfish website and place the executable in this directory.

Expected structure:

engines/stockfish/stockfish-windows-x86-64-avx2.exe

The Python scripts locate the Stockfish executable relative to the project directory, so no absolute local path is required.