| family | quant | device | task | acc | p50 ms | N |
|---|---|---|---|---:|---:|---:|
| Qwen3-1.7B | FP16 | iGPU | banking_cluster10 | 57.9% | 55.048572000032436 | 3080 |
| Qwen3-1.7B | FP16 | iGPU | clinc | 67.6% | 51.82415299987042 | 4500 |
| Qwen3-1.7B | FP16 | iGPU | cpu_routing | 69.0% | 54.27727200003574 | 58 |
| Qwen3-1.7B | FP16 | dGPU | banking_cluster10 | 57.9% | 10.497193999981391 | 3080 |
| Qwen3-1.7B | FP16 | dGPU | clinc | 67.6% | 10.425367999971513 | 4500 |
| Qwen3-1.7B | FP16 | dGPU | cpu_routing | 69.0% | 11.243533000026673 | 58 |
| Qwen3-1.7B | INT4 aggressive | iGPU | banking_cluster10 | 16.8% | 20.317835999776435 | 3080 |
| Qwen3-1.7B | INT4 aggressive | iGPU | clinc | 19.9% | 19.985397000255034 | 4500 |
| Qwen3-1.7B | INT4 aggressive | iGPU | cpu_routing | 34.5% | 20.520527999906335 | 58 |
| Qwen3-1.7B | INT4 aggressive | dGPU | banking_cluster10 | 16.8% | 11.741440000037073 | 3080 |
| Qwen3-1.7B | INT4 aggressive | dGPU | clinc | 19.9% | 11.783084000057897 | 4500 |
| Qwen3-1.7B | INT4 aggressive | dGPU | cpu_routing | 34.5% | 12.89293299998917 | 58 |
| Qwen3-1.7B | INT4 aggressive | NPU | banking_cluster10 | 12.6% | 159.73831499923108 | 3080 |
| Qwen3-1.7B | INT4 aggressive | NPU | clinc | 1.5% | 158.9214539999375 | 4500 |
| Qwen3-1.7B | INT4 aggressive | NPU | cpu_routing | 39.7% | 159.405420999974 | 58 |
| Qwen3-1.7B | INT4 g128 | iGPU | banking_cluster10 | 46.6% | 23.620335000032355 | 3080 |
| Qwen3-1.7B | INT4 g128 | iGPU | clinc | 44.9% | 23.15846600004079 | 4500 |
| Qwen3-1.7B | INT4 g128 | iGPU | cpu_routing | 46.6% | 21.68967600027827 | 58 |
| Qwen3-1.7B | INT4 g128 | dGPU | banking_cluster10 | 46.6% | 12.46055800004342 | 3080 |
| Qwen3-1.7B | INT4 g128 | dGPU | clinc | 44.8% | 12.424310000028527 | 4500 |
| Qwen3-1.7B | INT4 g128 | dGPU | cpu_routing | 46.6% | 13.422612999988814 | 58 |
| Qwen3-1.7B | INT4 g128+AWQ | iGPU | banking_cluster10 | 55.1% | 23.25184299996863 | 3080 |
| Qwen3-1.7B | INT4 g128+AWQ | iGPU | clinc | 54.0% | 23.19898399991871 | 4500 |
| Qwen3-1.7B | INT4 g128+AWQ | iGPU | cpu_routing | 46.6% | 21.580947000074957 | 58 |
| Qwen3-1.7B | INT4 g128+AWQ | dGPU | banking_cluster10 | 55.1% | 12.437022000085562 | 3080 |
| Qwen3-1.7B | INT4 g128+AWQ | dGPU | clinc | 54.1% | 12.32778500002496 | 4500 |
| Qwen3-1.7B | INT4 g128+AWQ | dGPU | cpu_routing | 46.6% | 13.439612999945894 | 58 |
| Qwen3-1.7B | INT4 g128sym | NPU | banking_cluster10 | 29.6% | 351.8224100007501 | 3080 |
| Qwen3-1.7B | INT4 g128sym | NPU | clinc | 16.4% | 349.5614330022363 | 4500 |
| Qwen3-1.7B | INT4 g128sym | NPU | cpu_routing | 56.9% | 351.76440799932607 | 58 |
| Qwen3-1.7B | INT4 g128sym r0.8 | NPU | banking_cluster10 | 48.2% | 330.41427900025155 | 3080 |
| Qwen3-1.7B | INT4 g128sym r0.8 | NPU | clinc | 54.8% | 330.2656409996416 | 4500 |
| Qwen3-1.7B | INT4 g128sym r0.8 | NPU | cpu_routing | 65.5% | 327.12412199907703 | 58 |
| Qwen3-1.7B | INT4 g128sym+AWQ | NPU | banking_cluster10 | 57.4% | 350.1922850009578 | 3080 |
| Qwen3-1.7B | INT4 g128sym+AWQ | NPU | clinc | 59.8% | 351.62357300032454 | 4500 |
| Qwen3-1.7B | INT4 g128sym+AWQ | NPU | cpu_routing | 36.2% | 353.20308200061845 | 58 |
| Qwen3-1.7B | INT8 | iGPU | banking_cluster10 | 57.2% | 29.967213999952946 | 3080 |
| Qwen3-1.7B | INT8 | iGPU | clinc | 67.9% | 29.908632000115176 | 4500 |
| Qwen3-1.7B | INT8 | iGPU | cpu_routing | 70.7% | 35.11220600012166 | 58 |
| Qwen3-1.7B | INT8 | dGPU | banking_cluster10 | 57.2% | 13.351340999975037 | 3080 |
| Qwen3-1.7B | INT8 | dGPU | clinc | 67.9% | 13.099501999931817 | 4500 |
| Qwen3-1.7B | INT8 | dGPU | cpu_routing | 70.7% | 14.440905000014936 | 58 |
| Qwen3-1.7B | INT8 | NPU | banking_cluster10 | 58.3% | 670.842165999602 | 3080 |
| Qwen3-1.7B | INT8 | NPU | clinc | 67.3% | 670.0916199999938 | 4500 |
| Qwen3-1.7B | INT8 | NPU | cpu_routing | 70.7% | 672.1622640000078 | 58 |
| Qwen3.5-2B | FP16 | iGPU | banking_cluster10 | 72.9% | 66.86543199998596 | 3080 |
| Qwen3.5-2B | FP16 | iGPU | clinc | 78.0% | 70.09825199997977 | 4500 |
| Qwen3.5-2B | FP16 | iGPU | cpu_routing | 60.3% | 66.84065699994335 | 58 |
| Qwen3.5-2B | FP16 | dGPU | banking_cluster10 | 72.9% | 19.437147999724402 | 3080 |
| Qwen3.5-2B | FP16 | dGPU | clinc | 78.0% | 20.803173000103925 | 4500 |
| Qwen3.5-2B | FP16 | dGPU | cpu_routing | 60.3% | 20.243964999963282 | 58 |
| Qwen3.5-2B | INT4 aggressive | iGPU | banking_cluster10 | 45.4% | 27.817473999903086 | 3080 |
| Qwen3.5-2B | INT4 aggressive | iGPU | clinc | 52.1% | 34.736833000124534 | 4500 |
| Qwen3.5-2B | INT4 aggressive | iGPU | cpu_routing | 37.9% | 28.20624600008159 | 58 |
| Qwen3.5-2B | INT4 aggressive | dGPU | banking_cluster10 | 45.9% | 21.81732400003966 | 3080 |
| Qwen3.5-2B | INT4 aggressive | dGPU | clinc | 52.8% | 25.17436399966755 | 4500 |
| Qwen3.5-2B | INT4 aggressive | dGPU | cpu_routing | 37.9% | 22.324431000015466 | 58 |
| Qwen3.5-2B | INT4 g128 | iGPU | banking_cluster10 | 70.2% | 30.821383999864338 | 3080 |
| Qwen3.5-2B | INT4 g128 | iGPU | clinc | 75.7% | 39.62986200008345 | 4500 |
| Qwen3.5-2B | INT4 g128 | iGPU | cpu_routing | 51.7% | 31.455945000061547 | 58 |
| Qwen3.5-2B | INT4 g128 | dGPU | banking_cluster10 | 70.3% | 24.176761000035185 | 3080 |
| Qwen3.5-2B | INT4 g128 | dGPU | clinc | 75.6% | 28.20283199980622 | 4500 |
| Qwen3.5-2B | INT4 g128 | dGPU | cpu_routing | 51.7% | 25.55140599997685 | 58 |
| Qwen3.5-2B | INT8 | iGPU | banking_cluster10 | 72.1% | 35.678769000014654 | 3080 |
| Qwen3.5-2B | INT8 | iGPU | clinc | 76.2% | 40.627888000017265 | 4500 |
| Qwen3.5-2B | INT8 | iGPU | cpu_routing | 69.0% | 39.486559000124544 | 58 |
| Qwen3.5-2B | INT8 | dGPU | banking_cluster10 | 72.2% | 23.781599999892933 | 3080 |
| Qwen3.5-2B | INT8 | dGPU | clinc | 76.0% | 27.587497999775223 | 4500 |
| Qwen3.5-2B | INT8 | dGPU | cpu_routing | 67.2% | 25.581161999980395 | 58 |