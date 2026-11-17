## Differentiable Rendering Extension

This fork enhances FoundationPose with test-time differentiable rendering optimization using NVIDIA `nvdiffrast`:

- **`diff_renderer.py`**: Pose optimizer parameterized in Lie algebra $\mathfrak{so}(3)$ (Rodrigues formula) and $\mathbb{R}^3$ translation. Optimizes depth (Smooth L1), antialiased silhouette (MSE), and photometric objectives with occlusion masking and MAP soft-anchor priors.
- **`augmentations.py`**: Sensor corruption benchmarks (40% synthetic occlusion, motion blur, sensor noise, low-light illumination, and combined corruptions).
- **`run_evaluation_pipeline.py`**: Evaluation script computing ADD, ADD-S, geodesic rotation error, translation error, Recall@0.1d, Recall@0.05d, 5°5cm, and AUC.
- **`results.txt`**: Benchmark results comparing FoundationPose with and without differentiable rendering under domain shifts.
