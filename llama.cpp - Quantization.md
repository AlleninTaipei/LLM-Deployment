# llama.cpp - Quantization

* [Source](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md#quantization): llama.cpp/examples/quantize/README.md

| Model | Measure      |    F16 |   Q4_0 |   Q4_1 |   Q5_0 |   Q5_1 |   Q8_0 |
|------:|--------------|-------:|-------:|-------:|-------:|-------:|-------:|
|    7B | perplexity   | 5.9066 | 6.1565 | 6.0912 | 5.9862 | 5.9481 | 5.9070 |
|    7B | file size    |  13.0G |   3.5G |   3.9G |   4.3G |   4.7G |   6.7G |
|    7B | ms/tok @ 4th |    127 |     55 |     54 |     76 |     83 |     72 |
|    7B | ms/tok @ 8th |    122 |     43 |     45 |     52 |     56 |     67 |
|    7B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
|   13B | perplexity   | 5.2543 | 5.3860 | 5.3608 | 5.2856 | 5.2706 | 5.2548 |
|   13B | file size    |  25.0G |   6.8G |   7.6G |   8.3G |   9.1G |    13G |
|   13B | ms/tok @ 4th |      - |    103 |    105 |    148 |    160 |    131 |
|   13B | ms/tok @ 8th |      - |     73 |     82 |     98 |    105 |    128 |
|   13B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |

## Summry

|Summary|trade-offs|
|-|-|
|Models|7B: A model with 7 billion parameters|
||13B: A model with 13 billion parameters|
|Quantization levels|F16: Full 16-bit precision (baseline)|
||Q4_0, Q4_1: 4-bit quantization (two variants)|
||Q5_0, Q5_1: 5-bit quantization (two variants)|
||Q8_0: 8-bit quantization|
|Measures|Perplexity: A measure of how well the model predicts text. Lower is better.|
||File size: The size of the model on disk in gigabytes (G).|
||ms/tok @ 4th: Milliseconds per token on 4th generation hardware.|
||ms/tok @ 8th: Milliseconds per token on 8th generation hardware.|
||Bits/weight: The number of bits used to represent each weight in the model.|

|Key observations|Description|
|-|-|
|Perplexity|The 13B model consistently has lower perplexity than the 7B model, indicating better performance. As quantization becomes more aggressive (fewer bits), perplexity slightly increases, showing a small trade-off in performance.|
|File size|Quantization significantly reduces file size. For example, the 7B model goes from 13.0G (F16) to 3.5G (Q4_0). The 13B model files are consistently larger than the 7B model files.|
|Speed (ms/tok)|Lower values indicate faster processing. 8th generation hardware is generally faster than 4th generation. More aggressive quantization (e.g., Q4_0, Q4_1) tends to be faster than less aggressive quantization. The 13B model is slower than the 7B model due to its larger size.|
|Bits/weight|This shows the compression level, ranging from 16 bits (F16) down to 4.5 bits (Q4_0). The bits/weight are consistent between the 7B and 13B models for each quantization level.|

## Q & A

|Q4_0 (4-bit quantization, type 0)|Q4_1 (4-bit quantization, type 1)|
|:------|:------|
|This is a simpler form of 4-bit quantization.|This is a slightly more sophisticated form of 4-bit quantization.|
|It uses 4 bits to represent each weight in the neural network.|It still uses 4 bits per weight, but with some additional complexity.|
|Typically, it uses a linear quantization scheme.|It may use a non-linear quantization scheme or a more advanced grouping method.|
|It may have a single scale factor for a group of weights.|Often, it includes an additional parameter (like a zero-point) to improve the representation of weights.|

|Key|differences based on the data|
|-|-|
|File size|Q4_1 results in slightly larger file sizes compared to Q4_0. For the 7B model, Q4_0 is 3.5G while Q4_1 is 3.9G.|
|Perplexity|Q4_1 shows slightly better perplexity compared to Q4_0, indicating potentially better model performance. For the 7B model, Q4_0 has a perplexity of 6.1565, while Q4_1 has 6.0912.|
|Speed|The speeds are very similar, with Q4_1 being marginally slower in some cases. For the 7B model on 4th gen hardware, Q4_0 takes 55 ms/tok while Q4_1 takes 54 ms/tok.|
|Bits/weight|Despite both being 4-bit quantization methods, Q4_1 uses 5.0 bits/weight compared to Q4_0's 4.5 bits/weight. This suggests that Q4_1 might be storing some additional information per weight group.|
|In general|Q4_1 seems to offer a slight improvement in model quality at the cost of a small increase in model size and potentially a minor impact on speed. The choice between Q4_0 and Q4_1 would depend on the specific requirements of the application, balancing factors like model size, performance, and speed.|