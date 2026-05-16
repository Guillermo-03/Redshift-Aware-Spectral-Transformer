# Data

Dataset files are downloaded at runtime via HuggingFace.

## Source

**Multimodal Universe — DESI**
- HuggingFace dataset: `MultimodalUniverse/desi`
- Survey: DESI DR1 / SV3 "one-percent" survey
- ~1M raw objects available via the HuggingFace preview split

## What I used

Only spectra with `ZWARN == 0` are used (reliable redshift pipeline fits). `ZWARN != 0` indicates a failed or ambiguous fit and those spectra are discarded.

After filtering, we select **200,000 clean spectra** (180K train / 20K val), shuffled with `seed=42` before splitting.

## Schema

Each example has:
- `spectrum['flux']` — float32 array of 7,781 flux measurements across 3,600–9,800 Å
- `spectrum['ivar']` — float32 inverse variance per pixel; `ivar == 0` marks bad/missing pixels
- `Z` — float32 redshift (ground truth label)
- `ZWARN` — int pipeline warning flag; we keep only `ZWARN == 0`

## Preprocessing (applied per spectrum at load time)

1. NaN/inf → 0
2. Bad pixels (`ivar == 0`) set to 0
3. Median/MAD normalization computed on valid pixels only: `(flux - median) / (MAD + 1e-8)`
4. Bad pixels reset to 0 after normalization
5. Zero-padded from 7,781 to 7,784 pixels (nearest multiple of 28)
6. Split into 278 non-overlapping patches of 28 pixels each

## Acknowledgments

Thanks to the DESI Collaboration for making their spectral data publicly available, and to the Multimodal Universe team for packaging it into a clean HuggingFace dataset — without that I'd have spent most of this project wrangling FITS files.

Dataset citation:
> The Multimodal Universe Collaboration, "The Multimodal Universe: Enabling Large-Scale Machine Learning with 76 Billion Astronomical Measurements", NeurIPS 2024 Datasets and Benchmarks Track.

---

## Colab cache

On Colab, the 200K clean examples are saved to Google Drive at:

```
MyDrive/RAST/desi_clean_200k/
```

This avoids re-scanning HuggingFace after a disconnect. If you change `n_train`, `n_val`, or the ZWARN filter, delete this directory and `latest.pt` before resuming — see CLAUDE.md for details.
