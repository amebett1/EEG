# Khung thuyết trình 10 phút — Single-trial P300 classification

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

## Slide 2 — One automated pipeline for 42 subjects

- **Người nói**: V2 – Thái Nguyễn Hoàng Bách (1:15)
- **Hình**: `report/figures/grand_average_39subs.png` + sơ đồ 7 bước (crop task → bad channels → Picard ICA + ICLabel → 0.5–40 Hz → epochs −200..1000 ms @ 250 Hz → labels).
- **Key takeaways**:
  - Same pipeline for all 42 subjects (MNE-Python), 97.0% of epochs kept.
  - ICA removed on average 15 components (mostly muscle, ~3 ocular); 3 blink components removed manually.
  - Behaviour recomputed from the markers matches the dataset paper exactly (RT 542.1 ± 65.5 ms).
  - 3 subjects misunderstood the task (77–99% false alarms to distractors) → excluded. Classification set: **39 subjects, 21,878 correct trials, 13.5% targets**.
- **Script**:
  > "We built one reproducible pipeline and ran it on all forty-two subjects: we cropped the recordings to the task, detected and interpolated bad channels, removed eye, heart and muscle components with ICA and ICLabel, filtered between 0.5 and 40 hertz and cut epochs from minus 200 to 1000 milliseconds. We kept 97 percent of the epochs. As a sanity check, the reaction times we recomputed from the markers match the dataset paper to the decimal. Three subjects pressed the button for almost every distractor — they misunderstood the task — so, like the original authors, we excluded them. That leaves 39 subjects and almost 22,000 trials, of which only 13.5 percent are targets."

## Slide 3 — The P300 is there: grand-average ERPs at Fz and Pz

- **Người nói**: V3 – Trương Huy Hoàng (1:15)
- **Hình**: `report/figures/grand_average_fz_pz.png`
- **Key takeaways**:
  - Pz, 300–500 ms: target 7.14 µV vs standard 1.51 µV → **+5.64 ± 2.12 µV**, t(38) = 16.6, p = 4.9 × 10⁻¹⁹, **Cohen's d = 2.66**.
  - Parietal maximum (Pz > Fz) in **39/39** subjects.
  - Larger P3b ↔ faster responses: r = −0.43, p = 0.006.
- **Script**:
  > "Before classifying anything, we checked that the P300 is really there. At Pz, targets produce a positivity of about seven microvolts between 300 and 500 milliseconds, against one and a half for standards. That is a difference of more than five microvolts, with an effect size of 2.7 — enormous for EEG — and it is maximal over the parietal cortex in every one of the 39 subjects. Subjects with a larger P300 also responded faster, which fits the idea that the P300 indexes the resources allocated to the stimulus."

## Slide 4 — P3a versus P3b, as predicted by Polich (2007)

- **Người nói**: V3 – Trương Huy Hoàng (0:45)
- **Hình**: `report/figures/sub-01_difference_waves.png` và `report/figures/population_p300_summary.png` (lưu ý: hình topomap hiện là của sub-01, ghi rõ trên slide).
- **Key takeaways**:
  - Target − standard (P3b): later, parietal. Distractor − standard (P3a): earlier, fronto-central.
  - All four Polich criteria → P300 verified in 39/39 normal subjects, 0/3 task-misunderstood subjects.
- **Script**:
  > "Difference waves separate the two subcomponents described by Polich: the distractor produces an earlier, more frontal P3a — an orienting response — while the target produces the later, parietal P3b linked to updating working memory. Across all criteria, the P300 is verified in every normal subject and in none of the three who misunderstood the task. Kien will now show how we turned this into features."

## Slide 5 — From signal to 402 features (Laton et al., 2014)

- **Người nói**: V4 – Nguyễn Công Kiên (1:15)
- **Hình**: bảng nhóm đặc trưng (mục 6.3 của `REPORT_EN.md`) và bảng "best feature per group" (mục 6.4); có thể vẽ nhanh bar chart top-10 từ `results/tables/v4_feature_auc.csv`.
- **Key takeaways**:
  - 12 channels × 5 windows: window amplitudes, P300/N1/P2/N2 peaks (Laton et al., 2014), Hjorth, band power, spectral entropy, db4 wavelet → **402 features per trial**.
  - P3 window widened to **300–600 ms**: in 5/10 subjects the Pz peak came after 500 ms.
  - Best single feature: **mean Pz amplitude 400–600 ms (AUC 0.84)** — exactly the P3b.
- **Script**:
  > "Following Laton and colleagues, we described each single trial with 402 features from twelve channels: amplitudes in five time windows, peak amplitudes and latencies, Hjorth parameters, spectral power and wavelet energy. One practical lesson: a 300-to-500 millisecond window cut off the peak in half of the subjects, so we widened it to 600. The most discriminative single feature is simply the mean amplitude at Pz between 400 and 600 milliseconds — the P3b itself — with an AUC of 0.84 on its own."

## Slide 6 — RQ1: how well do standard machine-learning models do?

- **Người nói**: V5 – Nguyễn Việt Đức (1:15)
- **Hình**: `report/figures/baseline_model_comparison.png` (phụ: `baseline_pr_curves.png`)
- **Key takeaways**:
  - Leakage-free: scaler inside the pipeline, **GroupKFold by subject** → test subjects never seen in training.
  - Imbalance (13.5% targets): class weights; balanced accuracy, ROC-AUC, PR-AUC instead of raw accuracy.
  - Logistic regression: **balanced accuracy 87.4%**; shrinkage LDA: **ROC-AUC 0.947**; PR-AUC ≈ 0.79 (≈ 6 × prevalence).
  - Random forest ranks well (AUC 0.91) but its sensitivity collapses to 29% at the default threshold.
- **Script**:
  > "Two things make this problem tricky: targets are only 13 percent of trials, so accuracy is meaningless — always saying 'standard' gives 87 percent — and trials from the same person must never be in both training and test sets. So we used class weights, balanced metrics, and five-fold cross-validation split by subject. Regularised linear models win: logistic regression reaches 87 percent balanced accuracy, and LDA with shrinkage — the classic P300 BCI method — an AUC of 0.95. This is our machine-learning answer to RQ1, and the benchmark for deep learning."

## Slide 7 — RQ2: does deep learning beat machine learning? Yes.

- **Người nói**: V6 – Đồng Minh Đức (1:30)
- **Hình**: `report/figures/ml_vs_dl_comparison.png` (phụ: `eegnet_roc_pr_real.png`)
- **Key takeaways**:
  - EEGNet-8,2 on raw epochs (128 × 301), 3,930 parameters, **same 39 subjects and identical folds** as the baselines.
  - **Balanced accuracy 90.7 ± 2.1%, ROC-AUC 0.970 ± 0.010, PR-AUC 0.873 ± 0.042** — best on every metric except specificity.
  - Paired per subject: higher AUC in **35/39 (vs LDA)** and **34/39 (vs logistic regression)**, Wilcoxon p < 10⁻⁸.
  - Why: all 128 channels and learned spatial filters vs 12 channels; full time course vs 5 window means; enough data (~15,000 trials per fold). On one subject only, EEGNet reached just AUC 0.89.
- **Script**:
  > "We then trained EEGNet, a compact convolutional network with fewer than four thousand parameters, directly on the raw epochs — and crucially, on exactly the same subjects and the same folds as the baselines. EEGNet is the best model overall: about 91 percent balanced accuracy, an AUC of 0.97 and a PR-AUC of 0.87. The gain in AUC is moderate, but it is consistent: EEGNet is better in 35 of 39 subjects. Why? It uses all 128 channels and learns spatial filters, and it sees the full time course instead of five window averages. But this only works with enough data: trained on a single subject, it was clearly worse. So for RQ2: yes, on this dataset deep learning wins — given enough data."

## Slide 8 — RQ3: what did EEGNet learn? + conclusion

- **Người nói**: V6 – Đồng Minh Đức (1:30), kết bài V1 – Nguyễn Văn Vũ (0:15)
- **Hình**: `report/figures/eegnet_saliency_topomap_real.png`
- **Key takeaways**:
  - **Time — consistent with the P3b**: 40% of saliency within 300–600 ms (window = 25% of the epoch), peak at 504 ms; almost nothing before stimulus onset.
  - **Space — only partly**: P300 effect is maximal at Pz, but saliency peaks over **left central sites (C1, FCC3h, C3)**, near 460 ms — just before the button press (RT ≈ 540 ms). Pz ranks 37/128; r = 0.32.
  - Caveat: gradients show the model's filters, not where the signal is (Haufe et al., 2014). Next step: EEGNet on distractor vs standard (no button press).
  - **Answer to RQ3: partly interpretable** — the network uses the P3b window, but may also exploit motor preparation.
- **Script**:
  > "Is it a black box? We computed saliency maps — how much each channel and time point influences the target decision. In time, the answer is reassuring: the network looks mostly at the P3b window, peaking around 500 milliseconds, and ignores the baseline. In space, it is more surprising: the P300 is maximal at Pz, but the network's attention is strongest over left central electrodes — the motor cortex — just before the button press. Saliency shows the network's filters, not the source of the signal, so this could be noise cancellation; but it could also mean that part of the deep-learning advantage comes from motor preparation rather than the P300. The decisive test is to train EEGNet on distractors versus standards, where nobody presses a button. So our answer to RQ3 is: partly interpretable."
  >
  > **V1 (kết bài):** "To conclude: the P300 is robust in all 39 subjects; single trials can be classified across subjects with up to 91 percent balanced accuracy; deep learning outperforms standard machine learning on this dataset, but understanding exactly what it uses requires more than a saliency map. Thank you."

---

## Chuẩn bị câu hỏi (Q&A)

| Câu hỏi có thể gặp | Người trả lời | Ý chính |
|---|---|---|
| Is the comparison fair? | V6 | Same 39 subjects, same 21,878 trials, identical subject-wise folds (verified per subject), same metrics and 0.5 threshold. |
| Isn't the classifier just detecting the button press? | V6 / V2 | Possible, especially for EEGNet (left-central saliency). Feature-based distractor vs standard (no press) still gives AUC 0.90 vs 0.95. Planned control: EEGNet on distractor vs standard. |
| Why balanced accuracy and not accuracy? | V5 | 13.5% targets: an all-standard classifier has 86.5% accuracy and detects nothing. |
| Why do LDA and logistic regression differ in balanced accuracy but not AUC? | V5 | Same ranking quality (AUC 0.947 vs 0.946); different operating point at threshold 0.5 because of class weighting. |
| Why 300–600 ms for features but 300–500 ms for ERP verification? | V4 / V3 | 300–500 ms follows the course criterion; for features, 5/10 subjects peaked after 500 ms, and 300–600 ms follows the dataset paper and Laton et al. |
| Would EEGNet be better with more training? | V6 | Possibly: 4/5 folds hit the 50-epoch limit; no hyper-parameter tuning for either approach. |
| Why exclude three subjects? | V2 | Task misunderstood (77–99% false alarms to distractors), also excluded by the dataset authors; their P300 fails the verification. |
