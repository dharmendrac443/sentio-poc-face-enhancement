# Sentio POC Face Enhancement

This project enhances low-resolution CCTV face crops using a CPU-only classical computer vision pipeline. It extracts unique face crops from a video, upsamples and sharpens them to `240x240`, then generates a comparison report and evaluation metrics.

## What The Project Does

The repository has two main scripts:

1. `extract.py`
   Extracts up to 100 unique face crops from a source video into `raw_faces/`.
2. `solution.py`
   Enhances every face in `raw_faces/`, saves outputs to `enhanced_faces/`, generates `enhancement_report.html`, and writes `evaluation_metrics.json`.

Reference images in `reference_identities/` are used for evaluation only.

## Project Structure

```text
sentio-poc-face-enhancement/
├── Demo_video.mov
├── extract.py
├── solution.py
├── requirements.txt
├── raw_faces/
├── enhanced_faces/
├── reference_identities/
├── enhancement_report.html
└── evaluation_metrics.json
```

## Enhancement Pipeline

`solution.py` applies the following stages:

1. Denoising with `cv2.fastNlMeansDenoisingColored`
2. CLAHE enhancement on the LAB luminance channel
3. Multi-step Lanczos upscaling to exactly `240x240`
4. Region-aware sharpening using MediaPipe Face Mesh landmarks

The pipeline also computes:

- recognition accuracy before enhancement
- recognition accuracy after enhancement
- Laplacian sharpness before and after enhancement
- SSIM similarity score
- per-image metrics in `evaluation_metrics.json`

## Requirements

Install dependencies from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current dependencies:

- `numpy==1.26.4`
- `opencv-contrib-python==4.13.0.92`
- `Pillow==10.3.0`
- `ImageHash==4.3.1`
- `scikit-image==0.22.0`
- `face_recognition==1.3.0`
- `mediapipe==0.10.14`

## How To Run

Run all commands from the repository root:

```bash
cd /data/dharmendra/Learning/sentio-poc-face-enhancement
```

### 1. Extract face crops from the demo video

`extract.py` defaults to `video.mov`, but this repository includes `Demo_video.mov`, so use:

```bash
SENTIO_VIDEO_PATH=Demo_video.mov python3 extract.py
```

This script:

- samples every 15th frame
- detects faces with OpenCV Haar cascades
- expands each crop for context
- removes duplicates using perceptual hashing
- saves up to 100 crops into `raw_faces/`

### 2. Run the enhancement pipeline

```bash
python3 solution.py
```

This reads input images from `raw_faces/` and writes enhanced results to `enhanced_faces/`.

## Generated Outputs

After running `solution.py`, the following outputs are generated:

- `enhanced_faces/` with enhanced `240x240` images
- `enhancement_report.html` with side-by-side raw and enhanced comparisons
- `evaluation_metrics.json` with summary metrics and per-face details

## Current Checked-In Results

The current `evaluation_metrics.json` in this repository reports:

- `total_faces_processed`: `100`
- `processing_time_sec`: `30.68`
- `recognition_accuracy_before_pct`: `68.0`
- `recognition_accuracy_after_pct`: `59.0`
- `avg_sharpness_before`: `112.7`
- `avg_sharpness_after`: `70.7`
- `avg_ssim_improvement`: `0.691`

These values come from the current checked-in dataset and can change if you rerun extraction or enhancement.

## Notes

- Run `solution.py` from the project root because it uses relative paths.
- `enhancement_report.html` uses relative image paths such as `raw_faces/...` and `enhanced_faces/...`, so keep the HTML file in the project root.
- If `mediapipe` is unavailable, the sharpening stage falls back to uniform unsharp masking.
- If `face_recognition` is unavailable, recognition metrics will not be populated meaningfully.
- `reference_identities/` is only for evaluation and matching, not for enhancement.
