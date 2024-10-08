# llama-finetune.exe Command Help

| **llama-finetune.exe**          | **Description**                                                                                              | **Default**                    |
|---------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------|
| `-h, --help`                    | Show help message and exit                                                                                   |                                |
| `--model-base FNAME`            | Path to the base model                                                                                       | `''`                           |
| `--lora-out FNAME`              | Path to save LLaMA LoRA                                                                                      | `ggml-lora-ITERATION-f32.gguf` |
| `--only-write-lora`             | Only save LoRA, no training                                                                                  |                                |
| `--norm-rms-eps F`              | RMS-Norm epsilon value                                                                                       | `0.000010`                     |
| `--rope-freq-base F`            | Frequency base for ROPE                                                                                      | `10000.000000`                 |
| `--rope-freq-scale F`           | Frequency scale for ROPE                                                                                     | `1.000000`                     |
| `--lora-alpha N`                | LoRA alpha for scaling                                                                                       | `4`                            |
| `--lora-r N`                    | Default rank for LoRA                                                                                        | `4`                            |
| `--rank-att-norm N`             | Rank for attention norm tensor                                                                               |                                |
| `--rank-ffn-norm N`             | Rank for feed-forward norm tensor                                                                            |                                |
| `--rank-out-norm N`             | Rank for output norm tensor                                                                                  |                                |
| `--rank-tok-embd N`             | Rank for token embeddings tensor                                                                             |                                |
| `--rank-out N`                  | Rank for output tensor                                                                                       |                                |
| `--rank-wq N`                   | Rank for `wq` tensor                                                                                         |                                |
| `--rank-wk N`                   | Rank for `wk` tensor                                                                                         |                                |
| `--rank-wv N`                   | Rank for `wv` tensor                                                                                         |                                |
| `--rank-wo N`                   | Rank for `wo` tensor                                                                                         |                                |
| `--rank-ffn-gate N`             | Rank for `ffn_gate` tensor                                                                                   |                                |
| `--rank-ffn-down N`             | Rank for `ffn_down` tensor                                                                                   |                                |
| `--rank-ffn-up N`               | Rank for `ffn_up` tensor                                                                                     |                                |
| `--train-data FNAME`            | Path to the training data                                                                                    | `shakespeare.txt`              |
| `--checkpoint-in FNAME`         | Path to load training checkpoint                                                                             | `checkpoint.gguf`              |
| `--checkpoint-out FNAME`        | Path to save training checkpoint                                                                             | `checkpoint-ITERATION.gguf`    |
| `--pattern-fn-it STR`           | Pattern in output filenames to replace with iteration number                                                 | `ITERATION`                    |
| `--fn-latest STR`               | String to use instead of iteration number for saving latest output                                           | `LATEST`                       |
| `--save-every N`                | Save checkpoint and LoRA every N iterations                                                                  | `10`                           |
| `-s SEED, --seed SEED`          | Random seed                                                                                                  | `-1`                           |
| `-c N, --ctx N`                 | Context size during training                                                                                 | `128`                          |
| `-t N, --threads N`             | Number of threads                                                                                            | `6`                            |
| `-b N, --batch N`               | Parallel batch size                                                                                          | `8`                            |
| `--grad-acc N`                  | Number of gradient accumulation steps                                                                        | `1`                            |
| `--sample-start STR`            | Starting point for samples after specified pattern                                                           | `''`                           |
| `--include-sample-start`        | Include sample start in samples                                                                              | `off`                          |
| `--escape`                      | Process sample start escape sequences                                                                        |                                |
| `--overlapping-samples`         | Allow overlapping samples                                                                                    | `off`                          |
| `--fill-with-next-samples`      | Follow short samples with next shuffled samples                                                              | `off`                          |
| `--separate-with-eos`           | Insert end-of-sequence token between samples                                                                 |                                |
| `--separate-with-bos`           | Insert begin-of-sequence token between samples                                                               |                                |
| `--no-separate-with-eos`        | Do not insert end-of-sequence token between samples                                                          |                                |
| `--no-separate-with-bos`        | Do not insert begin-of-sequence token between samples                                                        |                                |
| `--sample-random-offsets`       | Use samples beginning at random offsets                                                                      |                                |
| `--force-reshuffle`             | Force reshuffling of data at program start                                                                   |                                |
| `--no-flash`                    | Do not use flash attention                                                                                   |                                |
| `--use-flash`                   | Use flash attention                                                                                          |                                |
| `--no-checkpointing`            | Do not use gradient checkpointing                                                                             |                                |
| `--use-checkpointing`           | Use gradient checkpointing                                                                                   |                                |
| `--warmup N`                    | Number of warmup steps (Adam optimizer)                                                                      | `100`                          |
| `--cos-decay-steps N`           | Number of cosine decay steps (Adam optimizer)                                                                | `1000`                         |
| `--cos-decay-restart N`         | Increase of cosine decay steps after restart (Adam optimizer)                                                | `1.100000`                     |
| `--cos-decay-min N`             | Cosine decay minimum (Adam optimizer)                                                                        | `0.100000`                     |
| `--enable-restart N`            | Enable restarts of cosine decay                                                                              |                                |
| `--disable-restart N`           | Disable restarts of cosine decay                                                                             |                                |
| `--opt-past N`                  | Number of optimization iterations to track for delta convergence test                                        | `0`                            |
| `--opt-delta N`                 | Maximum delta for delta convergence test                                                                     | `0.000010`                     |
| `--opt-max-no-improvement N`    | Maximum number of optimization iterations with no improvement                                                | `0`                            |
| `--epochs N`                    | Maximum number of epochs to process                                                                          | `-1`                           |
| `--adam-iter N`                 | Maximum number of Adam optimization iterations per batch                                                     | `256`                          |
| `--adam-alpha N`                | Adam learning rate alpha                                                                                     | `0.001000`                     |
| `--adam-min-alpha N`            | Minimum learning rate alpha including warmup phase                                                           | `0.000000`                     |
| `--adam-decay N`                | AdamW weight decay                                                                                           | `0.100000`                     |
| `--adam-decay-min-ndim N`       | Minimum tensor dimensions for applying weight decay                                                          | `2`                            |
| `--adam-beta1 N`                | AdamW beta1                                                                                                  | `0.900000`                     |
| `--adam-beta2 N`                | AdamW beta2                                                                                                  | `0.999000`                     |
| `--adam-gclip N`                | AdamW gradient clipping                                                                                      | `1.000000`                     |
| `--adam-epsf N`                 | AdamW epsilon for convergence test                                                                           | `0.000000`                     |
| `-ngl N, --n-gpu-layers N`      | Number of model layers to offload to GPU                                                                     | `0`                            |
