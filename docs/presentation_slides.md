# Khung thuyết trình nháp (8 slide) & Ngân hàng Q&A — Single-trial P300 classification

> **Lưu ý:** Bộ slide bảo vệ chính thức (15 slide LaTeX Beamer 16:9) và kịch bản phân bổ thời lượng mới nhất nằm tại [`slides/main.pdf`](../slides/main.pdf), [`slides/main.tex`](../slides/main.tex) và [`slides/README.md`](../slides/README.md). File này được giữ lại làm khung tham khảo ban đầu và ngân hàng câu hỏi Q&A mở rộng ở cuối trang.

Môn AIT3012 Neuroscience (GS. Guy Nagels, VUB). Thời lượng tối đa **10 phút** (Slide 144) → khung dưới đây dự kiến **9 phút 45 giây**, chừa 15 giây dự phòng.

- 8 slide, mỗi slide có: tiêu đề, hình minh họa (trong `report/figures/`), thông điệp chính (Key takeaways), kịch bản nói bằng tiếng Anh và người nói.
- Toàn bộ số liệu lấy từ `report/REPORT_EN.md` (39 người, 21.878 trial, `GroupKFold` 5 fold theo người).
- Cả bài xoay quanh **3 câu hỏi nghiên cứu** của thầy: RQ1 ở slide 6, RQ2 ở slide 7, RQ3 ở slide 8.

| # | Slide | Người nói | Thời lượng |
|:---:|---|---|:---:|
| 1 | Research questions & dataset | V1 – Nguyễn Văn Vũ | 0:45 |
| 2 | Preprocessing pipeline & data quality | V2 – Thái Nguyễn Hoàng Bách | 1:15 |
| 3 | The P300 is there: grand-average ERPs | V3 – Trương Huy Hoàng | 1:15 |
| 4 | P3a vs P3b: topography & 42-subject verification | V3 – Trương Huy Hoàng | 0:45 |
| 5 | From signal to features (Laton et al., 2014) | V4 – Nguyễn Công Kiên | 1:15 |
| 6 | RQ1: machine-learning baselines | V5 – Nguyễn Việt Đức | 1:15 |
| 7 | RQ2: EEGNet vs machine learning | V6 – Đồng Minh Đức | 1:30 |
| 8 | RQ3: what did EEGNet learn? + conclusion | V6 – Đồng Minh Đức, kết bài V1 | 1:30 + 0:15 |
| | **Tổng** | | **9:45** |

---

## Slide 1 — Can we detect a P300 on a single trial?

- **Người nói**: V1 – Nguyễn Văn Vũ (0:45)
- **Hình**: không có hình dữ liệu; sơ đồ đơn giản của oddball task (standard 75% / target 12% / distractor 12%) và 3 câu hỏi nghiên cứu.
- **Key takeaways**:
  - Nencki-Symfonia visual oddball, 42 healthy adults, 128-channel EEG (GigaDB 100990).
  - Single-trial classification is what a P300 BCI needs, not averages.
  - RQ1: best single-trial standard-vs-target performance? RQ2: is deep learning better than standard ML? RQ3: is what deep learning learns interpretable?
- **Script**:
  > "Good morning. The P300 is one of the most robust brain responses we know, but in practice — in a brain–computer interface — we have to detect it on a single trial, not in an average of hundreds. We used the Nencki-Symfonia visual oddball dataset: forty-two healthy adults, 128-channel EEG, with frequent standards, rare targets that require a button press, and rare distractors. We asked three questions: how well can we classify single trials, does deep learning beat standard machine learning, and can we understand what the deep network learned. Bach will start with the data."

## Slide 2 — One standardised pipeline for 42 subjects

- **Người nói**: V2 – Thái Nguyễn Hoàng Bách (1:15)
- **Hình**: `report/figures/grand_average_39subs.png` + sơ đồ 7 bước (crop task → bad channels → Picard ICA + ICLabel → 0.5–40 Hz → epochs −200..1000 ms @ 250 Hz → labels).
- **Key takeaways**:
  - Same pipeline for all 42 subjects (MNE-Python), 97.0% of epochs kept.
  - ICA removed on average 15 components (mostly muscle, ~3 ocular); 3 blink components removed manually.
  - Target RT from task QC agrees with the dataset paper (542.1 ± 65.5 ms); the supplied epoch/feature hand-off passed the implemented audit.
  - 3 subjects misunderstood the task (77–99% false alarms to distractors) → excluded. Classification set: **39 subjects, 21,878 correct trials, 13.5% targets**.
- **Script**:
  > "We built one reproducible pipeline and ran it on all forty-two subjects: we cropped the recordings to the task, detected and interpolated bad channels, removed eye, heart and muscle components with ICA and ICLabel, filtered between 0.5 and 40 hertz and cut epochs from minus 200 to 1000 milliseconds. We kept 97 percent of the epochs. As a sanity check, the reaction times we recomputed from the markers match the dataset paper to the decimal. Three subjects pressed the button for almost every distractor — they misunderstood the task — so, like the original authors, we excluded them. That leaves 39 subjects and almost 22,000 trials, of which only 13.5 percent are targets."

## Slide 3 — Group-level P300: grand-average ERPs at Fz and Pz

- **Người nói**: V3 – Trương Huy Hoàng (1:15)
- **Hình**: `report/figures/grand_average_fz_pz.png`
- **Key takeaways**:
  - Correct trials only, **equal subject weights**, 39 subjects; SEM across subjects.
  - Pz, 300–500 ms: target **6.97 ± 2.74 µV**, standard **1.92 ± 1.49 µV**.
  - Paired difference **5.05 ± 2.28 µV**; **t(38) = 13.82, p = 2.14 × 10⁻¹⁶, dz = 2.21**.
  - Target Pz > Fz: **39/39**; RT–target-amplitude correlation: **r = −0.493, p = 0.00144**.
- **Script**:
  > "We averaged correct trials within each person and then weighted all 39 people equally. Between 300 and 500 milliseconds, targets produce about seven microvolts at Pz, compared with about two for standards. The paired difference is five microvolts, with an effect size of 2.2. Every included subject has a larger target response at Pz than Fz. Larger Pz responses are associated with faster reaction times, although this correlation does not establish causation."

## Slide 4 — Group topography and individual variability

- **Người nói**: V3 – Trương Huy Hoàng (0:45)
- **Hình**: `report/figures/grand_average_p300_summary_topomap.png`; phụ: `grand_average_difference_waves.png`.
- **Key takeaways**:
  - Group maps are based on **39 people**, not a sub-01 example.
  - Search target Pz maxima in **200–800 ms**; test acceptance **300–500 ms** independently.
  - **22/39** meet all three project-defined core P3b checks; **32/39** in secondary 300–600 ms.
  - Relative frontal distractor tendency: **26/39**; not uniform across subjects.
  - Three exclusions are based on task behaviour; ERP checks do not reselect the ML/DL cohort.
- **Script**:
  > "The group maps show a parietal target effect, but individual timing varies. We searched for each person's peak over 200 to 800 milliseconds before testing the narrower window. Twenty-two of 39 subjects met all three operational P3b checks; widening the analysis to 600 milliseconds gave 32. These are descriptive project criteria, not diagnostic rules. Task behaviour determined the three exclusions; the classification cohort stays unchanged."

## Slide 5 — From signal to 402 features (Laton et al., 2014)

- **Người nói**: V4 – Nguyễn Công Kiên (1:15)
- **Hình**: bảng nhóm đặc trưng và top đặc trưng (Bảng 1 & Bảng 5 trong `docs/EEG_Final_Report_EN.pdf`, mục 3 của `REPORT_EN.md`); có thể vẽ nhanh bar chart top-10 từ `results/tables/v4_feature_auc.csv`.
- **Key takeaways**:
  - 12 channels × 5 windows: window amplitudes, P300/N1/P2/N2 peaks (Laton et al., 2014), Hjorth, band power, spectral entropy, db4 wavelet → **402 features per trial**.
  - P3 window widened to **300–600 ms**: in **17/39 subjects (43.6%)** the Pz peak came after 500 ms.
  - Best single feature: **mean Pz amplitude 400–600 ms (AUC 0.84)** — exactly the P3b.
- **Script**:
  > "Following Laton and colleagues, we described each single trial with 402 features from twelve channels: amplitudes in five time windows, peak amplitudes and latencies, Hjorth parameters, spectral power and wavelet energy. One practical lesson: a 300-to-500 millisecond window cut off the peak in half of the subjects, so we widened it to 600. The most discriminative single feature is simply the mean amplitude at Pz between 400 and 600 milliseconds — the P3b itself — with an AUC of 0.84 on its own."

## Slide 6 — RQ1: how well do standard machine-learning models do?

- **Người nói**: V5 – Nguyễn Việt Đức (1:15)
- **Hình**: `report/figures/baseline_model_comparison.png` (phụ: `baseline_pr_curves.png`)
- **Key takeaways**:
  - Leakage-free: scaler inside the pipeline, **GroupKFold by subject** → test subjects never seen in training.
  - Imbalance (13.5% targets): class weights; balanced accuracy, ROC-AUC, PR-AUC instead of raw accuracy.
  - Logistic regression: **balanced accuracy 87.5%**; shrinkage LDA: **ROC-AUC 0.947**; PR-AUC ≈ 0.79 (≈ 6 × prevalence).
  - Random forest ranks well (AUC 0.91) but its sensitivity collapses to 29% at the default threshold.
- **Script**:
  > "Two things make this problem tricky: targets are only 13 percent of trials, so accuracy is meaningless — always saying 'standard' gives 87 percent — and trials from the same person must never be in both training and test sets. So we used class weights, balanced metrics, and five-fold cross-validation split by subject. Regularised linear models win: logistic regression reaches 87 percent balanced accuracy, and LDA with shrinkage — the classic P300 BCI method — an AUC of 0.95. This is our machine-learning answer to RQ1, and the benchmark for deep learning."

## Slide 7 — RQ2: does deep learning beat machine learning? Yes.

- **Người nói**: V6 – Đồng Minh Đức (1:30)
- **Hình**: `report/figures/ml_vs_dl_comparison.png`, phụ `lda_channel_controls_roc.png`
- **Key takeaways**:
  - EEGNet-8,2: 128 × 301 preprocessed samples, 3,930 parameters; same 39 participants and five outer test groups.
  - Balanced accuracy **90.3 ± 2.1%**, ROC-AUC **0.970 ± 0.013**, AP **0.865 ± 0.045**.
  - LDA given all 128 channels and the full epoch in 100 ms bins reaches AUC **0.949**. Electrode count alone does not explain EEGNet's lead.
  - Temporal representations and optimisation still differ; the result concerns the tested pipelines.
- **Script**:
  > "EEGNet achieves 90.3 percent balanced accuracy and an AUC of 0.970 on unseen participants. We also gave LDA all 128 channels over the full epoch. It reached AUC 0.949, so electrode count alone does not explain the advantage. The temporal representations still differ, so our conclusion is about these implemented pipelines: EEGNet performs best."



## Slide 8 — RQ3: what did EEGNet learn? + conclusion

- **Người nói**: V6 – Đồng Minh Đức (1:30), kết bài V1 – Nguyễn Văn Vũ (0:15)
- **Hình**: `report/figures/eegnet_controls_oof.png`, phụ `eegnet_controls_topomaps.png`
- **Key takeaways**:
  - All explanations use positive trials from each model's **outer test fold**. Fold maps are normalised before pooling.
  - Target saliency: peak **504 ms**, **40.7%** mass in 300–600 ms, spatial **r = 0.253**; Pz ranks **43/128**.
  - Completed no-button distractor–standard control: AUC **0.926 ± 0.015**.
  - RQ3: partly interpretable. Rare-stimulus discrimination is possible without a button-press contrast; target-specific motor contributions remain unquantified because stimuli and relevance also differ.
- **Script**:
  > "We now explain every outer test fold rather than training examples. Target sensitivity peaks at 504 milliseconds, with 40.7 percent in the P300 interval, but spatial correspondence is only partial. We also completed the no-button control: distractor versus standard reaches AUC 0.926. EEGNet can therefore discriminate these stimuli without a motor-response contrast. This does not isolate the target model's motor contribution, because stimulus identity and task relevance also differ. Our answer is partial interpretability, supported by held-out explanations and a completed behavioural control."
  >
  > **V1:** "The group parietal response is strong; EEGNet reaches 90.3 percent balanced accuracy on unseen participants and leads the tested baselines. Test-only explanations and no-button decoding strengthen the stimulus-related evidence while leaving the exact mechanism open. Thank you."

---



## Chuẩn bị câu hỏi (Q&A)

| Câu hỏi có thể gặp | Người trả lời | Ý chính |
|---|---|---|
| Is the comparison fair? | V6 | Same 39 subjects, same 21,878 trials, identical subject-wise folds (verified per subject), same metrics and 0.5 threshold. |
| Isn't the classifier just detecting the button press? | V6 / V2 | Target saliency does peak over left-central sites (C1, FCC3h, C3) near the 542 ms mean RT, consistent with motor-related sensitivity; however, the completed no-button distractor–standard control still reaches AUC 0.926 with an anterior shift (AFz, AFF2h, FC1), showing stimulus discrimination does not require a button-press contrast while leaving target-specific motor shares unresolved. |
| Why is spatial r computed over the full epoch rather than 300–600 ms? | V6 | Full-epoch mean saliency correlates at r = 0.253 (target) and 0.102 (distractor); restricting saliency to the 300–600 ms window shown in Figure 7 gives r = 0.203 and 0.048 — supporting the same conclusion of partial spatial correspondence. |
| Why balanced accuracy and not accuracy? | V5 | 13.5% targets: an all-standard classifier has 86.5% accuracy and detects nothing. |
| Why do LDA and logistic regression differ in balanced accuracy but not AUC? | V5 | Same ranking quality (AUC 0.947 vs 0.946); different operating point at threshold 0.5 because of class weighting. |
| Why 300–600 ms for features but 300–500 ms for ERP verification? | V4 / V3 | 300–500 ms is the project’s narrower operational window, not a mandatory course rule. Peaks are searched independently over 200–800 ms; a secondary 300–600 ms ERP analysis shows the effect of the window. No participant is reselected for ML/DL based on ERP checks. |
| Would EEGNet be better with more training? | V6 | Possibly: 4/5 folds hit the 50-epoch limit; no hyper-parameter tuning for either approach. |
| Why exclude three subjects? | V2 | Task misunderstood (77–99% false alarms to distractors), also excluded by the dataset authors. They can still have an ERP response; exclusion is behavioural, not proof of absent P300. Their correct distractor averages use only 1, 1 and 19 trials. |
