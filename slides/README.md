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
| **3** | **Neurophysiological foundation: dual-process P300 model** | **Trương Huy Hoàng** (V3 ERP) | **1:15 -- 2:00** | Polich (2007) dual-process theory: frontal P3a (involuntary orienting) vs parietal P3b (context updating); difference waveforms isolating cognitive potentials. |
| **4** | **Empirical verification of P300 morphology** | **Trương Huy Hoàng** (V3 ERP) | **2:00 -- 2:45** | Large oddball effect ($\Delta = 5.05\,\mu\text{V}, p < 10^{-18}$, Cohen's $d = 2.66$); 100% parietal dominance across 39 subjects; RT correlation ($r = -0.493$). |
| **5** | **Preprocessing pipeline and participant screening** | **Thái Nguyễn Hoàng Bách** (V2 Preprocessing) | **2:45 -- 3:30** | Picard ICA with automated ICLabel artifact removal; discovery and exclusion of 3 task-misunderstood participants (>90% distractor false alarms). |
| **6** | **Spatial and temporal dynamics of target processing** | **Thái Nguyễn Hoàng Bách** (V2 Preprocessing) | **3:30 -- 4:15** | 128-channel topomap evolution; parietal P3b focus peaking across 350--550 ms; sensory P100/N100 at occipital sites. |
| **7** | **Classical machine learning on 402 engineered features** | **Nguyễn Công Kiên** (V4) / **Nguyễn Việt Đức** (V5) | **4:15 -- 5:15** | Laton et al. peaks, db4 wavelet sub-bands, Hjorth activity; 5-fold participant-wise GroupKFold; Shrinkage LDA achieves ROC-AUC 0.947, Logistic Regression achieves Bal. Acc 0.875. |
| **8** | **End-to-end deep learning with compact EEGNet architecture** | **Đồng Minh Đức** (V6 Deep Learning) | **5:15 -- 6:00** | End-to-end 2D depthwise-separable CNN on raw 128-channel epochs; 2,370 parameters; cosine annealing with early stopping; achieves ROC-AUC 0.970, PR-AUC 0.865. |
| **9** | **Research question 1: best single-trial classification** | **Đồng Minh Đức** (V6 Deep Learning) | **6:00 -- 7:00** | **RQ1 Answer**: EEGNet yields best overall discrimination (ROC-AUC 0.970, Bal. Acc 0.892). **RQ2 Answer**: DL modestly outperforms ML (+0.023 AUC, +0.017 Bal. Acc), driven by end-to-end spatio-temporal filter synthesis. |
| **10** | **Research question 2: deep learning versus machine learning** | **Đồng Minh Đức** (V6 Deep Learning) | **7:00 -- 7:45** | Matched 128-channel LDA reaches ROC-AUC 0.949; demonstrates that channel density accounts for +0.002 AUC, while architectural filter synthesis provides the remaining +0.021 AUC. |
| **11** | **Distractor control: isolating P300 from motor response** | **Đồng Minh Đức** (V6 Deep Learning) | **7:45 -- 8:30** | No-button distractor control (Distractor vs Standard, 21,748 trials); EEGNet maintains ROC-AUC 0.926 ± 0.015 without motor confounds, proving authentic P300 decoding. |
| **12** | **RQ3: spatio-temporal saliency reveals true cognitive mechanisms** | **Đồng Minh Đức** (V6 Deep Learning) | **8:30 -- 9:15** | **RQ3 Answer**: Saliency gradient peaks at 504 ms (40.7% energy in 300--600 ms window); spatial topomap shows parietal P3b focus; sensorimotor activation vanishes in distractor condition. |
| **13** | **Summary answering the three core research questions** | **Nguyễn Văn Vũ** (V1 Lead) | **9:15 -- 9:40** | Synthesis table directly answering RQ1, RQ2, and RQ3 with empirical evidence and theoretical justification. |
| **14** | **Team contributions aligned with course roles** | **Nguyễn Văn Vũ** (V1 Lead) | **9:40 -- 9:55** | Explicit breakdown of 6 roles (V1 to V6) matching Slide 143 specifications; verified equal 100.00% total contribution. |
| **15** | **Conclusion** | **Nguyễn Văn Vũ** (V1 Lead) | **9:55 -- 10:00** | Final closing and thank you to the evaluation committee and audience. |

---

## 4. Answering Committee Questions (Defense Cheat Sheet)

1. **Why is the deep learning advantage over LDA modest (+0.023 ROC-AUC)?**
   * *Answer*: The P300 event-related potential is fundamentally a phase-locked, low-frequency, largely linear spatio-temporal dipole pattern across the parietal scalp. Linear shrinkage LDA with Ledoit-Wolf regularization already extracts the vast majority of separable variance. EEGNet's gain comes from learning continuous temporal and spatial filter kernels directly from raw voltages rather than discrete extracted peaks, consistent with findings by Lotte et al. (2018).

2. **How do you prove that EEGNet does not merely classify the button press?**
   * *Answer*: We implemented an explicit no-button distractor control (Distractor vs Standard, 21,748 trials), where *neither* stimulus involves any motor action. EEGNet still achieves ROC-AUC 0.926 ± 0.015, confirming it genuinely decodes the cognitive P300 waveform. Furthermore, input-gradient saliency topomaps reveal that the left-central sensorimotor cluster present in Target trials completely vanishes in Distractor trials.

3. **How was data leakage prevented across folds?**
   * *Answer*: We enforced strict 5-fold participant-wise `GroupKFold` cross-validation across all 39 compliant participants. Test participants were completely isolated from training and validation folds. All feature scaling (`StandardScaler` and global tensor variance normalization) was fitted solely inside the training folds.
