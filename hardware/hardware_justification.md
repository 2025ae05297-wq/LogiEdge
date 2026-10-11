# Hardware Selection

The dominant Constraint Triangle vertex is **reliability/latency under a strict power ceiling**, with cost the decisive secondary constraint at fleet scale. The chosen Raspberry Pi 5 plus AI HAT+ costs about ₹15,000/truck and draws 7.5 W, staying below the 10 W AI budget while providing ample accelerator headroom for a six-feature MLP and local MQTT/Docker services. The 90-second requirement is therefore met without a network dependency.

| Option | Strength | Limitation | Decision |
|---|---|---|---|
| Pi 5 + HAT+ | 13 TOPS, 7.5 W, ₹15k; good Linux/container support | More power and cost than an MCU | **Deploy** |
| Jetson Orin Nano | 67 TOPS and mature CUDA ecosystem | 15 W exceeds the AI budget; ₹45k is ₹2.55M for 85 trucks | Reject for this pilot |
| STM32H7 custom MCU | ₹3,500 and 0.4 W | Limited memory/tooling; risky for MQTT, containers, and future models | Reject as primary node |

Pilot hardware cost is ₹1.275M for 85 Pi nodes versus ₹3.825M for Jetson. At 265 vehicles, Pi is ₹3.975M versus ₹11.925M. The MCU is cheaper, but the engineering and operational risk is not justified for a safety-critical first deployment.

## Roofline calculation

Arithmetic intensity is `45 MFLOPs / 18 MB = 2.5 FLOPs/byte`. The Pi CPU ridge point is `16 GFLOP/s / 12 GB/s = 1.33 FLOPs/byte`. Since 2.5 exceeds 1.33, the stated model is compute-bound on the CPU roofline, with an ideal compute time of 2.81 ms versus a bandwidth time of 1.50 ms. INT8 quantisation and accelerator execution reduce compute and memory traffic; structured pruning reduces both further.
