# EEG P300 Single-Trial Classification - Slide Presentation Deck

This folder contains the complete, ready-to-present LaTeX Beamer slide deck and speaker script for the 10-minute final project defense in course **AIT3012 Neuroscience** (Instructor: **Prof. Guy Nagels**, VUB; Teaching Assistants: **Mr. Trần Quốc Sáng**, **Mr. Nguyễn Tiến Dũng**).

---

## 1. Directory Structure

```text
slides/
├── main.tex              # Master LaTeX Beamer presentation file (aspect ratio 16:9, 15 slides)
├── main.pdf              # Pre-compiled presentation PDF (15 slides)
├── README.md             # Compilation instructions & 10-minute speaker script
└── images/               # High-resolution logos & report figures
    ├── logo_uet.png      # VNU-UET emblem
    ├── logo_ai.png       # Institute of Artificial Intelligence emblem
    ├── v2_preprocessing_pipeline.png          # Fig 1 in report
    ├── grand_average_fz_pz.png                 # Fig 2 in report
    ├── grand_average_p300_summary_topomap.png  # Fig 3 in report
    ├── grand_average_difference_waves.png      # Fig 4 in report
    ├── baseline_roc_curves.png                 # Fig 5 in report
    ├── eegnet_roc_pr_real.png                  # Fig 6 in report
    ├── lda_channel_controls_roc.png            # Fig 7 in report
    ├── eegnet_controls_oof.png                 # Fig 8 in report
    └── eegnet_controls_topomaps.png            # Fig 9 in report
```

> **Note on figures**: All figures in `slides/images/` originate directly from the final report (`docs/EEG_Final_Report_EN.pdf`, Figures 1--9) alongside the two official university emblems. No external or redundant graphic assets are included.

---

## 2. Compilation Instructions

### Local Compilation (pdfLaTeX)
```powershell
cd slides
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Overleaf Compilation
1. Create a zip archive of the `slides/` directory (`main.tex` and `images/`).
2. Upload to [Overleaf](https://www.overleaf.com/).
3. In Overleaf Settings, verify compiler is set to **pdfLaTeX**.
4. Click **Recompile**.

---

## 3. 10-Minute Presentation Script & Slide Allocation (15 Slides)

The deck is paced for a strict 10-minute defense (Slide 144 requirement; **9:45 planned**, 15 s buffer), following the report's exact figure sequence (Figures 1--9) and aligning each speaker with their own project role (V1--V6):

| Slide | Title | Figure / Table | Speaker / Role | Target Time | Key Talking Points |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **Title slide** | Logos | **Nguyễn Văn Vũ** (V1 Lead) | **0:00 -- 0:25** | Project introduction, academic context (AIT3012 Neuroscience, Prof. Guy Nagels), and team overview. |
| **2** | **Research context and the three core questions** | — | **Nguyễn Văn Vũ** (V1 Lead) | **0:25 -- 1:00** | Nencki-Symfonia visual oddball (42 subjects, 128 ch), single-trial challenge, and RQ1/RQ2/RQ3 from Slide 142. |
| **3** | **Standardized preprocessing pipeline and quality control** | Fig. 1 | **Thái Nguyễn Hoàng Bách** (V2) | **1:00 -- 2:00** | Order: crop + FCz + bad-channel interpolation + average ref → Picard ICA + ICLabel ($p \ge 0.8$, +3 manual blink ICs) → 0.5--40 Hz FIR + apply ICA → epochs 250 Hz, 97.0% retention (27,034/27,865). Exclude 3 task-misunderstood subjects (76.5--98.8% distractor FA, mean 91.0%) → 21,878 correct binary trials (39 subjects, 13.49% targets). |
| **4** | **Group event-related potentials and parietal effect** | Fig. 2 | **Trương Huy Hoàng** (V3) | **2:00 -- 2:45** | Equal-participant weights (N=39, correct trials): target Pz $6.97 \pm 2.74\,\mu\text{V}$ vs standard $1.92 \pm 1.49\,\mu\text{V}$; $\Delta = 5.05 \pm 2.28\,\mu\text{V}$, $t(38) = 13.82$, $p = 2.14 \times 10^{-16}$, $d_z = 2.21$; 39/39 $Pz > Fz$; RT ($542.1 \pm 65.5$ ms) correlation $r = -0.493, p = 0.00144$. Note: red marker (8.8 µV, 480 ms) is the grand-average within-window peak. |
| **5** | **Group scalp topography and individual variability** | Fig. 3 / Tab. 4 | **Trương Huy Hoàng** (V3) | **2:45 -- 3:25** | 128-channel mean scalp maps (300--500 ms) show parietal maxima for both target and distractor contrasts; operational P300 checks (39/39 $Pz > Fz$, 37/39 oddball $\ge 1.5\,\mu\text{V}$, 22/39 peak within 300--500 ms vs 32/39 in 300--600 ms; 17/39 late peaks 508--748 ms). |
| **6** | **P300 framework (Polich, 2007) and difference waves** | Fig. 4 | **Trương Huy Hoàng** (V3) | **3:25 -- 4:00** | Polich (2007): easy target--standard discrimination (5.6% misses, 0.27% standard FA) yields a parietal distractor P3 ($+3.54\,\mu\text{V}$ at Pz, $-1.06\,\mu\text{V}$ at Fz) rather than a canonical frontal P3a, alongside parietal target P3b ($+5.05\,\mu\text{V}$) and a relative anterior distractor shift in 26/39 participants ($+1.43 \pm 2.46\,\mu\text{V}$). |
| **7** | **Conventional ML baseline with 402 features** | Fig. 5 / Tab. 5 | **Nguyễn Công Kiên** (V4) | **4:00 -- 4:50** | 402 features across 12 channels (120 window mean/SD, 36 P300 peak/lat/area, 18 N1/P2/N2 peaks, 12 ptp, 48 db4 wavelet, 36 Hjorth, 132 Welch spectral); strongest single feature `Pz_p3b_mean` (400--600 ms, AUC 0.836); 5-fold GroupKFold + inside-fold StandardScaler; pooled OOF ROC $\approx 0.942$ for LDA/LR/SVM. |
| **8** | **Compact deep learning architecture: EEGNet-8,2** | Fig. 6 / Tab. 3 | **Đồng Minh Đức** (V6) | **4:50 -- 5:35** | End-to-end 2D depthwise-separable CNN on preprocessed $128 \times 301$ epochs (250 Hz); 3,930 trainable parameters; max-norm 1.0/0.25, dropout 0.5; Adam ($\text{lr}=10^{-3}$), class-weighted CE, 15% trial-stratified inner val, patience 15 (max 50 epochs); pooled OOF ROC 0.968, AP 0.853. |
| **9** | **Research question 1: best single-trial classification** | Tab. 6 | **Nguyễn Việt Đức** (V5) | **5:35 -- 6:20** | **RQ1 Answer**: EEGNet leads overall (ROC-AUC $0.970 \pm 0.013$, AP $0.865 \pm 0.045$, Bal. Acc $0.903 \pm 0.021$, sens. 0.880, spec. 0.927), ahead of Shrinkage LDA (ROC-AUC 0.947) and Logistic Regression (Bal. Acc 0.875; +2.82 pp). Explain threshold-0.5 sensitivity drop for LDA (65.9%), SVM (62.5%), RF (28.9%) under 13.5% prevalence. |
| **10** | **Research question 2: deep learning vs. machine learning** | Fig. 7 / Tab. 8 | **Nguyễn Việt Đức** (V5) | **6:20 -- 7:05** | **RQ2 Answer (Yes, for tested pipelines)**: +0.023 ROC-AUC over 402-feature LDA (exploratory Wilcoxon $p = 5.0 \times 10^{-10}$, 35/39 higher) and +0.074 AP. Matched 128-ch binned LDA reaches 0.949 (+0.012 over 12-ch binned 0.937), leaving +0.021 AUC (+0.043 AP) for EEGNet; modest gain reflects P300's low-dimensional, largely linear spatio-temporal pattern (Blankertz et al., 2011; Lotte et al., 2018). |
| **11** | **No-button distractor control and outer-test profiles** | Fig. 8 / Tab. 10 | **Đồng Minh Đức** (V6) | **7:05 -- 7:50** | No-button control (21,748 correct distractor vs standard trials, identical outer folds): ROC-AUC $0.926 \pm 0.015$, Bal. Acc $0.855 \pm 0.020$, AP $0.745 \pm 0.048$. Walk through all 3 panels of Fig. 8 (pooled ROC 0.968 vs 0.925, late temporal saliency peaks 504 vs 524 ms, channel-wise $r = 0.253$ vs $0.102$). Emphasize scope caveat: proves rare-stimulus decoding without button press, does not isolate target motor share. |
| **12** | **Research question 3: outer-test gradient saliency topomaps** | Fig. 9 / Tab. 11 | **Đồng Minh Đức** (V6) | **7:50 -- 8:35** | **RQ3 Answer (Partly interpretable)**: Target outer-test saliency (504 ms, 40.7% in 300--600 ms, $r = 0.253$ / $0.203$ in window, Pz rank 43/128) highlights left-central sites (C1, FCC3h, C3) + Oz/P2. Distractors (524 ms, 43.9%, $r = 0.102$, Pz rank 56/128) lose the motor cluster and shift to anterior sites (AFz/AFF2h/FC1/AF3, possible residual ocular signal) + Oz (~100 ms). Gradients measure sensitivity, not neural sources (Haufe et al., 2014). |
| **13** | **Summary of answers to the three research questions** | — | **Nguyễn Văn Vũ** (V1 Lead) | **8:35 -- 9:15** | Concise synthesis table directly answering RQ1, RQ2, and RQ3 with quantitative evidence and neuroscientific/ML rationale. |
| **14** | **Team roles and contribution shares** | — | **Nguyễn Văn Vũ** (V1 Lead) | **9:15 -- 9:35** | Individual responsibilities and contribution shares across 6 roles (V1--V6: five members 17.67%, V2 11.65%; exact 100.00% total). |
| **15** | **Conclusion** | — | **Nguyễn Văn Vũ** (V1 Lead) | **9:35 -- 9:45** | Final closing and thank you (15 s buffer before 10:00 limit). |

---

## 4. Answering Committee Questions (Defense Cheat Sheet)

1. **Why is the deep learning advantage over LDA modest (+0.023 ROC-AUC)?**
   * *Answer*: The P300 event-related potential is fundamentally a phase-locked, low-frequency, largely linear spatio-temporal pattern across the parietal scalp. Linear shrinkage LDA with Ledoit-Wolf regularization already extracts the vast majority of separable variance. EEGNet's gain comes from learning continuous temporal and spatial filter kernels directly from preprocessed epoch time series rather than discrete extracted peaks, consistent with findings by Lotte et al. (2018).

2. **Isn't EEGNet merely classifying the button press?**
   * *Answer*: Target saliency does concentrate over left-central sensorimotor sites (C1, FCC3h, C3) near the 542.1 ms mean reaction time, consistent with motor-related sensitivity. However, our completed no-button distractor control (Distractor vs Standard, 21,748 trials, where neither condition involves a button press) still reaches ROC-AUC $0.926 \pm 0.015$ and the left-central cluster vanishes in favour of anterior (AFz, AFF2h, FC1, AF3) and occipital (Oz) sites. This shows EEGNet can discriminate rare stimuli without a button-press contrast, while leaving the exact motor share in target decoding unresolved because stimulus identity, task relevance, and ERP morphology also differ.

3. **How was data leakage prevented across folds?**
   * *Answer*: We enforced strict 5-fold participant-wise `GroupKFold` cross-validation across all 39 compliant participants. Test participants were completely isolated from training and validation folds. All feature scaling (`StandardScaler` for LDA/LR/SVM and global training-subset standard-deviation scaling for EEGNet) was fitted solely inside the training folds.
