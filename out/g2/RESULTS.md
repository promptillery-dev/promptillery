# G2 Banking77 Results

Generated from `out/g2` artifacts on 2026-07-09. Cost fields are `n/a` because token trackers stored `estimated_cost: null`.

Hardware: 2x NVIDIA GeForce RTX 4090, driver 595.71.05, CUDA 13.2 reported by nvidia-smi, 24564 MiB each. Profiles were measured on `cuda:0` except FastText on CPU.

## Table 4 Summary

| Student | Type | Params | Held-out metric | Fid. | Teacher tokens | $/1K | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FastText | Reference | 100d ngrams | Acc 0.8240 | 0.7162 | 66,004 | n/a | 0.01 | 0.01 |
| ModernBERT base | Encoder | 149M | Acc 0.9185 | 0.7841 | 62,866 | n/a | 12.16 | 12.27 |
| Ettin encoder 150M | Encoder | 150M | Acc 0.9188 | 0.7844 | 63,259 | n/a | 11.99 | 12.23 |
| Ettin decoder 150M | Decoder SLM | 150M | Macro-F1 0.9092 | 0.7769 | 107,104 | n/a | 95.17 | 151.04 |
| RoBERTa base | Encoder | 125M | Acc 0.9325 | 0.7932 | 62,999 | n/a | 4.06 | 4.16 |
| Gemma3 270M | Decoder SLM | 270M | Macro-F1 0.9119 | 0.7808 | 110,043 | n/a | 100.46 | 149.02 |

## Cycle Readout
| Student | Metric | Cycle 1 | Cycle 5 | Cycle 10 | Cum. tokens @10 |
|---|---|---:|---:|---:|---:|
| FastText | accuracy | 0.8041 | 0.8076 | 0.8066 | 66,004 |
| ModernBERT base | accuracy | 0.9190 | 0.9210 | 0.9165 | 62,866 |
| Ettin encoder 150M | accuracy | 0.9055 | 0.9150 | 0.9180 | 63,259 |
| Ettin decoder 150M | macro_f1 | 0.9135 | 0.8950 | 0.8785 | 107,104 |
| RoBERTa base | accuracy | 0.9205 | 0.9325 | 0.9330 | 62,999 |
| Gemma3 270M | macro_f1 | 0.8996 | 0.9064 | 0.8988 | 110,043 |

## Per-cycle Teacher Tokens
| Student | Cycle 1 tokens | Cycle 5 tokens | Cycle 10 tokens | Cycle costs |
|---|---:|---:|---:|---|
| FastText | 7,461 | 7,236 | 0 | n/a |
| ModernBERT base | 7,132 | 7,064 | 0 | n/a |
| Ettin encoder 150M | 7,073 | 7,060 | 0 | n/a |
| Ettin decoder 150M | 12,010 | 11,879 | 0 | n/a |
| RoBERTa base | 7,180 | 7,015 | 0 | n/a |
| Gemma3 270M | 12,203 | 12,080 | 0 | n/a |

## Token Totals
- Teacher-test materialization: 1,639,979 tokens (1,624,664 input, 15,315 output).
- Completed training runs: 472,275 tokens.
- Dropped attempts: 0 tokens.
- Campaign total: 2,112,254 tokens.
- Estimated cost: n/a in artifacts; plan estimate for teacher-test materialization was about $5.

## Completion-only Masking Evidence
| Decoder | Log evidence |
|---|---|
| Ettin decoder | `completion-only loss: masked 134958/155199 prompt tokens (87.0%) in split test` |
| Gemma3 270M | `completion-only loss: masked 134958/155199 prompt tokens (87.0%) in split test` |

## Dropped Runs
- SmolLM3 (P1): CUDA OOM after original launch, expandable-segments retry, generation_batch_size 4 retry, and generation_batch_size 2 retry; dropped per P1 rule. Teacher tokens consumed: 0.
- Qwen3 4B LoRA (P2): CUDA OOM with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; dropped per P2 rule with no config edit. Teacher tokens consumed: 0.

## Deviations
- Used uv-managed Python 3.13 because uv selected Python 3.14 initially and the lock supports up to cp313.
- Installed the fasttext optional dependency via uv sync --extra fasttext after the first FastText config validation failed.
- Teacher test materialization first attempt stalled before writing rows; killed and relaunched once, then completed with 3,080 rows.
- Two bounded foreground diagnostic probes for ModernBERT/Ettin decoder were moved under logs/aborted_probes and excluded from completed-run analysis.
- SmolLM3 used the plan-allowed OOM fallback edits only: generation_batch_size 8 -> 4 -> 2, then was dropped after retry3 OOM.
- RoBERTa and Gemma logged aiohttp ClientSession cleanup exceptions during Python shutdown after final metrics and token_usage.json had been saved.
- Qwen3 was dropped after OOM with expandable CUDA segments, per P2 rule; no Qwen config edit was made.
- promptillery analyze out/g2 --metric accuracy and --metric macro_f1 failed on the mixed encoder/decoder parent directory because each metric is absent for the other architecture family; report values were extracted directly from metrics.json.
- FastText profiling required --model-path pointing at the run directory. Passing the .bin file measured latency but hit a profile save-path bug.
- Gemma profiling required an explicit PEFT load with tokenizer-length embedding resize before adapter load; the profile CLI auto-load path failed with a one-token embedding size mismatch.

