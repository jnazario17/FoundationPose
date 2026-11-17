====================================================================================================
FOUNDATIONPOSE EXTENSION: CAN DIFFERENTIABLE RENDERING IMPROVE 6D POSE ESTIMATION?
====================================================================================================

Author / Fork: jnazario17 / FoundationPose
Base Repository: NVlabs/FoundationPose (CVPR 2024 Highlight)

----------------------------------------------------------------------------------------------------
1. PROJECT OVERVIEW & RESEARCH OBJECTIVE
----------------------------------------------------------------------------------------------------
FoundationPose is a state-of-the-art foundation model for 6D object pose estimation and tracking.
While FoundationPose achieves remarkable accuracy on uncorrupted inputs by combining neural network 
pose refinement with contrastive score prediction, realistic robotics and computer vision applications 
frequently encounter domain shifts—such as heavy object occlusions, severe motion blur, sensor noise, 
and poor lighting conditions.

The objective of this project is to investigate:
    "Can analytical test-time Differentiable Rendering improve FoundationPose pose estimation, 
     especially under challenging corruptions and sensor domain shifts?"

To answer this question, we implemented a custom GPU-accelerated differentiable rendering optimizer, 
developed a data corruption benchmark pipeline, and conducted extensive comparative evaluations across 
multiple error metrics.

----------------------------------------------------------------------------------------------------
2. NEW CODE ADDITIONS & ARCHITECTURE
----------------------------------------------------------------------------------------------------
The following files have been introduced in this fork:

1. diff_renderer.py
   - Implements `DifferentiablePoseOptimizer` powered by NVIDIA nvdiffrast (GPU rasterization context).
   - Pose Parameterization: Formulated directly in Lie algebra so(3) rotation vectors (via Rodrigues 
     formula with continuous gradient flow at the identity) and translation vectors in R^3.
   - Multi-Objective Loss Formulation:
         L(R, t) = w_depth * SmoothL1(pred_depth, obs_depth) 
                 + w_sil   * MSE(pred_silhouette, obs_mask) 
                 + w_rgb   * L1(pred_rgb, obs_rgb) 
                 + L_reg
   - MAP Soft-Anchor Prior (L_reg): A quadratic prior anchoring updates to the neural network hypothesis, 
     preventing unconstrained gradient descent from drifting when sensor noise is present.
   - Occlusion-Aware Visibility Masking: Disregards occluded obstacle regions during depth and silhouette 
     backpropagation.
   - Optimization: Adam optimizer with Cosine Annealing learning rate schedule.

2. augmentations.py
   - Implements realistic environmental and sensor corruption functions:
       * apply_occlusion: Synthetic opaque obstacles blocking 40% of the bounding box.
       * apply_blur: 21x21 Gaussian kernel (sigma=8) simulating rapid camera/object motion.
       * apply_sensor_noise: Additive Gaussian noise (sigma=35) + 20% random depth sensor dropout.
       * apply_low_light: 0.25x brightness scaling with sensor shot noise.
       * apply_combined: Simultaneous severe occlusion (35%) and motion blur (15x15).

3. run_evaluation_pipeline.py
   - Master end-to-end benchmark script executing head-to-head comparisons across the trajectory.
   - Computes standard 6D pose estimation metrics:
       * ADD (Average Distance of Model Points)
       * ADD-S (Symmetric Average Distance)
       * Geodesic Rotation Error (degrees)
       * Translation Error (cm)
       * Recall @ 0.1d (strict threshold: ADD < 10% object diameter)
       * Recall @ 0.05d (fine threshold: ADD < 5% object diameter)
       * 5 deg, 5 cm Recall
       * Area Under the Curve (AUC of ADD from 0 to 10 cm)

4. test_diff_refine.py
   - Standalone verification script for rapid experimentation and debugging of differentiable rendering 
     gradient steps on single frames.

5. run_demo.py (Updated)
   - Updated with a `--display 0` flag for headless Docker/server execution, preventing OpenCV Qt display 
     crashes while preserving track visualization saves to debug/track_vis/.

6. results.txt
   - Raw output and benchmark logs detailing demo results, clean baseline metrics, and comparative evaluation.

----------------------------------------------------------------------------------------------------
3. EXPERIMENTAL SETUP
----------------------------------------------------------------------------------------------------
- Dataset: Kinect Driller Sequence (744 RGB-D frames with textured 3D CAD mesh).
- Target Object: Driller (Diameter: 0.2573 m = 25.73 cm).
  * 0.1d Threshold:  2.57 cm
  * 0.05d Threshold: 1.29 cm
- Environment: Official Docker container (`foundationpose`) with NVIDIA GPU acceleration 
  (RTX 2070 SUPER, CUDA 11.8).
- Model Checkpoints: Official pretrained Refiner (2023-10-28) and Scorer (2024-01-11).

----------------------------------------------------------------------------------------------------
4. MASTER TABULATION OF RESULTS
----------------------------------------------------------------------------------------------------
Comparative performance between Baseline FoundationPose (Without Differentiable Rendering, "No DR") 
and with Differentiable Rendering ("WITH DR") on identical augmented data:

====================================================================================================================
Experiment / Condition     | ADD (cm) | ADD-S(cm) | Rot (deg) | Trans(cm) | Rec@0.1d  | Rec@0.05d | 5° 5cm  | AUC   
====================================================================================================================
Clean Baseline             |     0.26 |      0.15 |      1.25 |      0.22 |    100.0% |    100.0% |  100.0% |  97.3%
--------------------------------------------------------------------------------------------------------------------
Occlusion (No DR)          |     4.81 |      0.98 |     75.32 |      3.82 |     50.0% |     50.0% |   16.7% |  55.8%
Occlusion (WITH DR)        |     4.86 |      1.00 |     75.28 |      3.89 |     50.0% |     50.0% |   16.7% |  55.3%
--------------------------------------------------------------------------------------------------------------------
Blur (No DR)               |     0.35 |      0.19 |      1.71 |      0.31 |    100.0% |    100.0% |  100.0% |  96.7%
Blur (WITH DR)             |     0.47 |      0.23 |      1.71 |      0.44 |    100.0% |    100.0% |  100.0% |  95.3%
--------------------------------------------------------------------------------------------------------------------
Noise (No DR)              |     0.39 |      0.21 |      1.43 |      0.35 |    100.0% |    100.0% |  100.0% |  96.1%
Noise (WITH DR)            |     0.44 |      0.22 |      1.37 |      0.41 |    100.0% |    100.0% |  100.0% |  95.7%
--------------------------------------------------------------------------------------------------------------------
Low-Light (No DR)          |     0.30 |      0.17 |      1.59 |      0.27 |    100.0% |    100.0% |  100.0% |  97.2%
Low-Light (WITH DR)        |     0.59 |      0.26 |      1.61 |      0.57 |    100.0% |    100.0% |  100.0% |  94.1%
--------------------------------------------------------------------------------------------------------------------
Combined (No DR)           |     2.11 |      0.58 |     24.85 |      1.56 |     83.3% |     66.7% |   58.3% |  80.4%
Combined (WITH DR)         |     2.19 |      0.58 |     24.82 |      1.62 |     83.3% |     75.0% |   58.3% |  79.5%
====================================================================================================================

----------------------------------------------------------------------------------------------------
5. HOW TO RUN THE EVALUATION
----------------------------------------------------------------------------------------------------
Execute within the official Docker container:
    docker run --gpus all --rm \
        -v "/path/to/FoundationPose:/workspace/FoundationPose" \
        -w /workspace/FoundationPose \
        foundationpose python run_evaluation_pipeline.py

----------------------------------------------------------------------------------------------------
6. CONCLUSION
----------------------------------------------------------------------------------------------------
tbd
