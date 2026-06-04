# Dog2Vec Sanity Check

This repo tests whether Dog2Vec embeddings can identify individual dogs across completely different videos.

## Research Question

Can Dog2Vec recognize the same dog from a new recording session/video, rather than memorizing a specific video or recording condition?

## Dataset

| Metric | Value |
|---|---:|
| Dogs | 43 |
| Total bark clips | 1418 |
| Train clips | 1092 |
| Test clips | 326 |
| Train videos | video1, video2, video3 |
| Test video | video4 |
| Embedding model | Dog2Vec |
| Embedding dimension | 768 |
| Dog2Vec layer | 9 |

## Scripts

| Script | Purpose |
|---|---|
| `01_create_manifest.py` | Creates dataset manifest |
| `02_extract_embeddings.py` | Extracts Dog2Vec embeddings |
| `03_raw_cosine.py` | Raw cosine profile baseline |
| `04_direct_svm.py` | Direct Linear SVM |
| `05_weighted_cosine.py` | SVM-weighted cosine profile |
| `06_sanity_checks.py` | Dataset sanity checks |
| `07_random_split_svm.py` | Random bark split SVM |
| `08_same_video_split_svm.py` | Same-video split SVM |
| `09_topk_retrieval.py` | Top-K profile retrieval |
| `10_umap_visualization.py` | UMAP visualization |
| `11_nearest_neighbor.py` | Nearest-neighbor bark retrieval |
| `12_leave_one_video_out.py` | Leave-one-video-out SVM |
| `13_confusion_pairs.py` | Most confused dog pairs |
| `14_per_dog_analysis.py` | Per-dog best/worst analysis |

## Main Results

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Train SVM sanity check | 100.00% | 100.00% |
| Direct SVM cross-video | 41.41% | 38.21% |
| Leave-one-video-out SVM | 39.64% | 34.51% |
| Random bark split SVM | 74.37% | 70.49% |
| Same-video split SVM | 81.46% | 77.36% |
| Raw cosine profile | 33.44% | 28.97% |
| Weighted cosine profile | 32.21% | 27.00% |
| Nearest-neighbor bark retrieval | 34.66% | 30.96% |

## Top-K Retrieval

| Metric | Accuracy |
|---|---:|
| Top-1 | 33.44% |
| Top-3 | 49.39% |
| Top-5 | 60.43% |

## Leave-One-Video-Out

| Held-out video | Accuracy | Macro F1 |
|---|---:|---:|
| video1 | 40.41% | 32.20% |
| video2 | 38.44% | 33.68% |
| video3 | 38.29% | 33.95% |
| video4 | 41.41% | 38.21% |
| Average | 39.64% | 34.51% |

## Most Confused Pairs

| True Dog | Predicted Dog | Count |
|---|---|---:|
| Sherpa | Sherpa_s_Day | 8 |
| 柴犬でんちゃん | 豆柴すみすみ | 7 |
| Queen_Chi | Cheeky_Chis | 5 |
| Sherpa | Zeusy_The_Talking_Pitty | 4 |
| Sonny_the_chihuahua | Cheeky_Chis | 4 |

## Best Dogs

| Dog | Accuracy |
|---|---:|
| MaisytheBarnHippo | 100.00% |
| Bounce_The_Pit_Bull | 100.00% |
| Nikko_Boy | 87.50% |
| Shadow_The_Sweetest_Husky | 85.71% |
| Pluto | 83.33% |

## Worst Dogs

| Dog | Accuracy |
|---|---:|
| Sherpa | 0.00% |
| K_eyush_The_Stunt_Dog | 0.00% |
| Ragnar_Pitbull | 0.00% |
| Percy_The_Pitbull | 0.00% |
| Brindle_Pitbull_Bulletproof | 0.00% |

## Key Findings

The pipeline is working correctly because the Direct SVM reaches 100% training accuracy.

Dog2Vec embeddings contain useful discriminative information because random bark split accuracy reaches 74.37% and same-video split accuracy reaches 81.46%.

However, cross-video performance drops to about 40%, showing that the learned representation does not generalize reliably across different recording sessions.

Raw cosine, weighted cosine, and nearest-neighbor retrieval all perform around 32–35%, suggesting profile averaging is not the main bottleneck.

Weighted cosine helped in the original 4-dog benchmark, but did not scale to the 43-dog benchmark.

## Final Conclusion

Dog2Vec can separate bark clips under random or same-video conditions, but it struggles to identify individual dogs across completely different videos.

The current evidence suggests that Dog2Vec embeddings encode dog-discriminative acoustic information, but not a robust video-invariant dog fingerprint.
