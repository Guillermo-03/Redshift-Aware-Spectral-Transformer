# Redshift-Aware Spectral Transformer

A transformer-based representation learning project for astrophysical spectra, focused on masked spectrum reconstruction and redshift prediction.

This project explores how modern sequence models can be adapted to one-dimensional astronomical spectra. The goal is to build a model that learns meaningful spectral structure directly from flux measurements, without relying on imaging data, photometry, magnitudes, or hand-engineered features.

---

## Overview

Astronomical spectra contain rich physical information about celestial objects. Emission lines, absorption lines, continuum shape, and spectral breaks can reveal an object's composition, class, motion, and redshift.

This project focuses on building a compact spectral foundation model that can:

1. Reconstruct missing regions of a spectrum.
2. Predict redshift from spectral context.
3. Learn general-purpose representations of astrophysical spectra.

Rather than treating redshift as a separate downstream prediction task, this project trains redshift prediction jointly with masked spectrum reconstruction. This encourages the model's internal representation to capture one of the most important physical signals in observational astronomy.

---

## Motivation

Large astrophysics foundation models, such as AION-1, aim to learn from massive scientific datasets across many modalities. While that broad approach is powerful, it can also spread model capacity across images, spectra, scalar metadata, and other inputs.

This project takes a narrower approach.

Instead of building a broad multimodal model, it focuses deeply on one problem: learning strong representations of astrophysical spectra.

The goal is to solve one of the core problems that models like AION-1 hoped to address: building a model that can understand astronomical observations in a generalizable way. By focusing on spectra alone, this project prioritizes depth over breadth and attempts to learn the physical structure of spectra more directly.

A key design choice is to make redshift central to the training process. Redshift is not treated as an afterthought or a separate prediction head added after representation learning. Instead, the model is trained so that redshift prediction helps shape the learned spectral representation from the beginning.

---

## Approach

This project combines ideas from:

- Transformer-based sequence modeling
- Encoder-decoder representation learning
- Masked autoencoding
- Self-attention over spectral tokens
- Joint reconstruction and regression objectives
- Redshift-aware spectral modeling

The model learns from partially masked spectra. During training, parts of the input spectrum are hidden, and the model must reconstruct the missing regions using the surrounding spectral context.

At the same time, the model predicts the redshift of the object. This encourages the learned representation to capture long-range relationships between spectral features, such as shifted emission and absorption lines.

---

## Why Transformers for Spectra?

Spectra are naturally sequential data. Each spectrum is a measurement of flux across wavelength, and important information is often distributed across distant regions of the sequence.

A single spectral feature may be ambiguous on its own. However, multiple features together can reveal meaningful astrophysical structure. For example, redshift is often inferred by identifying several emission or absorption lines that have shifted together.

Transformers are well-suited for this because self-attention allows different wavelength regions to communicate with each other. This makes it possible for the model to learn relationships between local features, broad continuum structure, and long-range spectral patterns.

---

## Redshift-Aware Learning

Redshift is one of the most important physical quantities encoded in an astronomical spectrum. It describes how much spectral features have shifted from their rest-frame wavelengths and is closely tied to distance and cosmic expansion.

This project trains the model to predict redshift jointly with masked spectrum reconstruction.