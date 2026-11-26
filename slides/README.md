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

The deck is specifically paced for a strict 10-minute project presentation (Slide 144 requirement):

| Slide | Title | Speaker / Role | Target Time | Key Talking Points |
|:---:|:---|:---:|:---:|:---|
| **1** | **Title Slide** | **Nguyễn Văn Vũ** (V1 Lead) | **0:00 -- 0:35** | Project introduction, academic context (AIT3012 Neuroscience, Prof. Guy Nagels), and presentation overview. |
| **2** | **Three research questions defined by course guidelines** | **Nguyễn Văn Vũ** (V1 Lead) | **0:35 -- 1:15** | State RQ1 (best performance), RQ2 (DL vs ML comparison), and RQ3 (explainability / failure analysis) from Slide 142. |
| **3** | **P300 framework (Polich, 2007) and difference waves** | **Trương Huy Hoàng** (V3 ERP) | **1:15 -- 2:00** | Polich (2007) framework: easy target--standard discrimination produces a parietal distractor P3 ($+3.54\,\mu\text{V}$ at Pz, $-1.06\,\mu\text{V}$ at Fz) rather than a canonical frontal P3a, alongside a strong parietal target P3b ($+5.05\,\mu\text{V}$) and a relative anterior distractor shift in 26/39 participants. |
| **4** | **Empirical verification of P300 morphology** | **Trương Huy Hoàng** (V3 ERP) | **2:00 -- 2:45** | Large oddball effect (target $6.97 \pm 2.74\,\mu\text{V}$ vs standard $1.92 \pm 1.49\,\mu\text{V}$; $\Delta = 5.05 \pm 2.28\,\mu\text{V}$, $t(38) = 13.82$, $p = 2.14 \times 10^{-16}$, $d_z = 2.21$); 39/39 parietal dominance ($Pz > Fz$); RT correlation ($r = -0.493, p = 0.00144$). |
| **5** | **Preprocessing pipeline and participant screening** | **Thái Nguyễn Hoàng Bách** (V2 Preprocessing) | **2:45 -- 3:30** | Bad-channel interpolation, average reference, Picard ICA + ICLabel ($p \ge 0.8$, +3 manual blink ICs), 0.5--40 Hz filter, 97.0% epoch retention (27,034/27,865); exclusion of 3 task-misunderstood participants (76.5--98.8% distractor false alarms, mean 91.0%); 21,878 correct binary trials (39 participants, 13.49% targets). |
| **6** | **Group scalp topography and individual variability** | **Thái Nguyễn Hoàng Bách** (V2 Preprocessing) | **3:30 -- 4:15** | 128-channel mean scalp maps (300--500 ms) show parietal maxima for both target and distractor contrasts; operational P300 checks (39/39 $Pz > Fz$, 37/39 oddball $\ge 1.5\,\mu\text{V}$, 22/39 peak within 300--500 ms vs 32/39 in 300--600 ms; 17/39 late peaks 508--748 ms). |
| **7** | **Classical machine learning on 402 engineered features** | **Nguyễn Công Kiên** (V4) / **Nguyễn Việt Đức** (V5) | **4:15 -- 5:15** | 402 features across 12 channels (120 window mean/SD, 36 P300 peak/lat/area, 18 N1/P2/N2 peaks, 12 peak-to-peak, 36 Hjorth, 132 Welch spectral, 48 db4 wavelet); 5-fold participant-wise GroupKFold; Shrinkage LDA achieves ROC-AUC 0.947, Logistic Regression achieves Bal. Acc 0.875. |
| **8** | **End-to-end deep learning with compact EEGNet architecture** | **Đồng Minh Đức** (V6 Deep Learning) | **5:15 -- 6:00** | End-to-end 2D depthwise-separable CNN on preprocessed 128-channel epochs; 3,930 trainable parameters; Adam ($\text{lr} = 10^{-3}$), class-weighted cross-entropy, early stopping (patience 15, max 50 epochs); achieves ROC-AUC 0.970, AP 0.865. |
| **9** | **Research question 1: best single-trial classification** | **Đồng Minh Đức** (V6 Deep Learning) | **6:00 -- 7:00** | **RQ1 Answer**: EEGNet yields best overall single-trial discrimination (ROC-AUC $0.970 \pm 0.013$, AP $0.865 \pm 0.045$, Bal. Acc $0.903 \pm 0.021$), ahead of Shrinkage LDA (ROC-AUC 0.947) and Logistic Regression (Bal. Acc 0.875; +2.82 pp for EEGNet). |
| **10** | **Research question 2: deep learning versus machine learning** | **Đồng Minh Đức** (V6 Deep Learning) | **7:00 -- 7:45** | Matched 128-channel binned LDA reaches ROC-AUC 0.949 vs 0.937 for 12-channel binned LDA (+0.012 AUC from electrode density), leaving a +0.021 ROC-AUC (+0.043 AP) lead for EEGNet; over 402-feature LDA: +0.023 ROC-AUC (exploratory Wilcoxon $p = 5.0 \times 10^{-10}$, 35/39 subjects) and +0.074 AP. |
| **11** | **No-button distractor control** | **Đồng Minh Đức** (V6 Deep Learning) | **7:45 -- 8:30** | No-button distractor control (Distractor vs Standard, 21,748 trials, matched outer folds); EEGNet reaches ROC-AUC $0.926 \pm 0.015$ (Bal. Acc $0.855 \pm 0.020$, AP $0.745 \pm 0.048$), showing rare-stimulus discrimination does not require a button press while leaving target motor share unquantified. |
| **12** | **RQ3: outer-test gradient saliency** | **Đồng Minh Đức** (V6 Deep Learning) | **8:30 -- 9:15** | **RQ3 Answer (Partly interpretable)**: Outer-test saliency peaks at 504 ms (40.7% mass in 300--600 ms; spatial $r = 0.253$, $0.203$ in 300--600 ms; Pz rank 43/128); target highlights left-central sites (C1, FCC3h, C3) + Oz/P2, whereas distractors (524 ms, 43.9%, $r = 0.102$, Pz rank 56/128) shift to anterior sites (AFz/AFF2h/FC1/AF3, possible residual ocular signal) and Oz (~100 ms); gradients measure sensitivity, not neural sources (Haufe et al., 2014). |
| **13** | **Summary answering the three core research questions** | **Nguyễn Văn Vũ** (V1 Lead) | **9:15 -- 9:40** | Evidence-based synthesis aligned with the report: RQ1 (EEGNet ROC-AUC 0.970, Bal. Acc 0.903), RQ2 (Yes, for tested pipelines; modest gain due to P300 linearity), RQ3 (Partly interpretable; stimulus decoding possible without button press, exact motor/neural mechanisms open). |
| **14** | **Team contributions aligned with course roles** | **Nguyễn Văn Vũ** (V1 Lead) | **9:40 -- 9:55** | Explicit breakdown of 6 roles (V1 to V6) matching Slide 143 specifications (five members 17.67%, V2 11.65%; exact 100.00% total). |
| **15** | **Conclusion** | **Nguyễn Văn Vũ** (V1 Lead) | **9:55 -- 10:00** | Final closing and thank you to the evaluation committee and audience. |

---

## 4. Answering Committee Questions (Defense Cheat Sheet)

1. **Why is the deep learning advantage over LDA modest (+0.023 ROC-AUC)?**
   * *Answer*: The P300 event-related potential is fundamentally a phase-locked, low-frequency, largely linear spatio-temporal pattern across the parietal scalp. Linear shrinkage LDA with Ledoit-Wolf regularization already extracts the vast majority of separable variance. EEGNet's gain comes from learning continuous temporal and spatial filter kernels directly from preprocessed epoch time series rather than discrete extracted peaks, consistent with findings by Lotte et al. (2018).

2. **Isn't EEGNet merely classifying the button press?**
   * *Answer*: Target saliency does concentrate over left-central sensorimotor sites (C1, FCC3h, C3) near the 542.1 ms mean reaction time, consistent with motor-related sensitivity. However, our completed no-button distractor control (Distractor vs Standard, 21,748 trials, where neither condition involves a button press) still reaches ROC-AUC $0.926 \pm 0.015$ and the left-central cluster vanishes in favour of anterior (AFz, AFF2h, FC1, AF3) and occipital (Oz) sites. This shows EEGNet can discriminate rare stimuli without a button-press contrast, while leaving the exact motor share in target decoding unresolved because stimulus identity, task relevance, and ERP morphology also differ.

3. **How was data leakage prevented across folds?**
   * *Answer*: We enforced strict 5-fold participant-wise `GroupKFold` cross-validation across all 39 compliant participants. Test participants were completely isolated from training and validation folds. All feature scaling (`StandardScaler` for LDA/LR/SVM and global training-subset standard-deviation scaling for EEGNet) was fitted solely inside the training folds.
