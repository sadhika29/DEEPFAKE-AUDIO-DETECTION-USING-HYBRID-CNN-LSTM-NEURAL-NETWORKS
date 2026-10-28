import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.dataset import generate_dfadd_demo_dataset
from src.preprocess import load_audio, extract_features_from_file, compute_acoustic_forensic_metrics
from src.predict import calculate_diffusion_forensic_score

def test_pipeline():
    print("=" * 65)
    print("VERIFYING DFADD AUDIO PIPELINE & DETECTION DIFFERENTIATION")
    print("=" * 65)

    # 1. Generate test samples
    manifest = generate_dfadd_demo_dataset(num_per_class=3)
    if hasattr(manifest, "to_dict"):
        records = manifest.to_dict(orient="records")
    else:
        records = list(manifest)

    print(f"Generated {len(records)} test samples in data/raw/")

    bona_files = [r["file_path"] for r in records if int(r["label"]) == 0]
    spoof_files = [r["file_path"] for r in records if int(r["label"]) == 1]

    print("\n--- 1. Testing Feature Extraction (Log-Mel Spectrogram) ---")
    spec_bona = extract_features_from_file(bona_files[0])
    spec_spoof = extract_features_from_file(spoof_files[0])
    print(f"Bona Fide Spectrogram: Shape = {spec_bona.shape} | Min = {spec_bona.min():.2f} | Max = {spec_bona.max():.2f}")
    print(f"Spoof Spectrogram:     Shape = {spec_spoof.shape} | Min = {spec_spoof.min():.2f} | Max = {spec_spoof.max():.2f}")
    assert spec_bona.shape == (128, 94, 1), f"Unexpected shape {spec_bona.shape}"

    print("\n--- 2. Testing Acoustic Forensic Differentiation ---")
    for i in range(len(bona_files)):
        y_b, sr = load_audio(bona_files[i])
        score_b, metrics_b = calculate_diffusion_forensic_score(y_b, sr)

        y_s, sr = load_audio(spoof_files[i])
        score_s, metrics_s = calculate_diffusion_forensic_score(y_s, sr)

        verdict_b = "Bona Fide (Authentic)" if score_b < 0.5 else "Spoof (Deepfake)"
        verdict_s = "Spoof (Deepfake)" if score_s >= 0.5 else "Bona Fide (Authentic)"

        print(f"\n[Test Pair #{i+1}]")
        print(f"  [AUTHENTIC VOICE]  File: {Path(bona_files[i]).name}")
        print(f"     Score: {score_b*100:.1f}% Spoof | Verdict: {verdict_b}")
        print(f"     Metrics: Centroid Var = {metrics_b['centroid_variance']:.1f} | HF Ratio = {metrics_b['high_frequency_ratio']:.3f}")
        print(f"  [DEEPFAKE VOICE]   File: {Path(spoof_files[i]).name}")
        print(f"     Score: {score_s*100:.1f}% Spoof | Verdict: {verdict_s}")
        print(f"     Metrics: Centroid Var = {metrics_s['centroid_variance']:.1f} | HF Ratio = {metrics_s['high_frequency_ratio']:.3f}")

        # Assert correct differentiation
        assert score_b < 0.45, f"Authentic sample got false high deepfake score: {score_b}"
        assert score_s >= 0.50, f"Deepfake sample got false low deepfake score: {score_s}"

    print("\n" + "=" * 65)
    print("SUCCESS: Pipeline produces accurate, differentiated results!")
    print("Authentic human voices are correctly identified as LOW RISK.")
    print("Diffusion deepfake voices are correctly identified as HIGH RISK.")
    print("=" * 65)

if __name__ == "__main__":
    test_pipeline()
