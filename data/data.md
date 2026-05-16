# Data

Dataset files are downloaded at runtime via HuggingFace.

## Source

**UniverseTBD — DESI EDR SV3**
- HuggingFace dataset: `UniverseTBD/mmu_desi_edr_sv3`
- Survey: DESI Early Data Release, SV3 "one-percent" survey
- 1,126,441 raw objects available

## What I used

Only spectra with `ZWARN == False` are used (reliable redshift pipeline fits). `ZWARN == True` indicates a failed or ambiguous fit and those spectra are discarded.

The dataset is streamed in 50K-row chunks, filtering each chunk for clean spectra, until 200,000 clean examples are collected. These are shuffled with `seed=42` before splitting into 180K train / 20K val.

## Schema

Each example has:
- `spectrum['flux']` — float32 array of 7,781 flux measurements across 3,600–9,800 Å
- `spectrum['ivar']` — float32 inverse variance per pixel; `ivar == 0` marks bad/missing pixels
- `spectrum['mask']` — boolean bad-pixel mask; `True` marks a bad pixel
- `Z` — float32 redshift (ground truth label)
- `ZWARN` — bool pipeline warning flag; we keep only `ZWARN == False`

## Preprocessing (applied per spectrum at load time)

1. NaN/inf → 0
2. Bad pixels (`ivar == 0` or `mask == True`) set to 0
3. Median/MAD normalization computed on valid pixels only: `(flux - median) / (MAD + 1e-8)`
4. Bad pixels reset to 0 after normalization
5. Zero-padded from 7,781 to 7,784 pixels (nearest multiple of 28)
6. Split into 278 non-overlapping patches of 28 pixels each

## Acknowledgments

Thanks to the DESI Collaboration for making their spectral data publicly available, and to UniverseTBD for packaging it into a clean HuggingFace dataset.

Dataset citation:
> The Multimodal Universe Collaboration, "The Multimodal Universe: Enabling Large-Scale Machine Learning with 76 Billion Astronomical Measurements", NeurIPS 2024 Datasets and Benchmarks Track.

---

## Colab cache

On Colab, the 200K clean examples are saved to Google Drive at:

```
MyDrive/RAST/desi_clean_200k/
```

This avoids re-scanning HuggingFace after a disconnect. If you change `n_train`, `n_val`, or the ZWARN filter, delete this directory and `latest.pt` before resuming — see CLAUDE.md for details.
