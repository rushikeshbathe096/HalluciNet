# HalluciNet: Training LLMs to Know the Limits of Their Own Knowledge

## The Problem

LLMs hallucinate. They generate confident, fluent, completely wrong statements.
This is the #1 unsolved problem in production AI deployment.

Incorrect confident outputs are more dangerous than uncertain ones —
a user cannot tell the difference. For a model serving 100 million users,
even a 3% hallucination rate means 3 million wrong confident answers per day.

No reinforcement learning environment existed to train agents to detect this.
We built the first one: HalluciNet Adversarial.

## The Two-Agent System

### Generator Agent
Receives a reference document. Creates a subtle hallucination designed
to fool the detector. Rewarded when the detector misses it.
Learns to create progressively harder hallucinations.

### Detector Agent
Receives reference + generated response. Must detect the hallucination.
Rewarded by a 4-dimension deterministic grader.
Learns to catch progressively subtler errors.

### Adversarial Self-Play Loop

```
Generator creates hallucination
          ↕
Detector tries to catch it
          ↕
Both get rewards based on outcome
          ↕
Curriculum escalates difficulty when both improve
          ↕
Recursive skill amplification — Theme 4
```

## The Unique Feature: Confidence Calibration

Most hallucination benchmarks only measure detection accuracy.
HalluciNet also rewards calibrated uncertainty:

| Outcome | Reward |
|---------|--------|
| Correct + High Confidence | +bonus |
| Correct + Low Confidence | base reward |
| Wrong + Low Confidence | small penalty |
| Wrong + High Confidence | severe penalty |

This trains agents to know when they are uncertain —
critical for production AI safety.

An overconfident wrong answer triggers no human review.
An uncertain wrong answer does.

**We are not training models to detect hallucinations.
We are training models to know the limits of their own knowledge.**

## Four Difficulty Levels

| Task | Samples | Design |
|------|---------|--------|
| easy | 8 | Obvious errors — wrong year, name, city |
| medium | 10 | Mixed errors — digit swaps, attribution |
| hard | 15 | Negation traps, entity confusion, adversarial clean |
| expert | 20 | Multi-hop reasoning, date arithmetic, numeric traps |

### Adversarial Clean Samples

The hard and expert tasks include counter-intuitive true facts
designed to trigger false positives:

- **Venus rotation** — "Venus day is longer than Venus year" sounds wrong but is true
- **Radioactive bananas** — sounds alarming but every figure is correct
- **Oxford vs Aztecs** — Oxford University predates the Aztec Empire by centuries

A naive detector flags these. Our trained agent learns to verify before flagging.

## Adaptive Curriculum — Theme 4: Self-Improvement

HalluciNet implements adaptive curricula:

- Agent starts on easy tasks
- Advances when detector catch rate consistently above 0.75 over 3 sessions
- Drops back when below 0.40
- Generator and detector push each other to their capability frontier permanently

This is recursive skill amplification — exactly Theme 4.

## Grader Design

| Component | Weight | Description |
|-----------|--------|-------------|
| Hallucination detection | 0.50 | Did the agent correctly identify existence? |
| Phrase identification | 0.30 | Did it quote the exact wrong phrase? |
| Correct fact | 0.20 | Did it provide the right correction? |
| Confidence calibration | ±0.10 | Was it appropriately certain? |

## Exploit Resistance

| Strategy | Score |
|----------|-------|
| Always-True agent | 0.300 |
| Always-False agent | 0.250 |
| Random agent | 0.573 |
| Correct calibrated agent | 0.999 |

No shortcut scores above 0.58.
Only genuine correct detection with calibrated confidence scores above 0.90.

## Real Adversarial Results

6 sessions with llama-3.1-8b-instant via Groq:

| Session | Task | Det Rate | Gen Rate | Decision |
|---------|------|----------|----------|----------|
| 1 | easy | 0.80 | 0.20 | stay |
| 2 | easy | 0.80 | 0.20 | stay |
| 3 | easy | 0.80 | 0.20 | promote → medium |
| 4 | medium | 0.80 | 0.20 | stay |
| 5 | medium | 0.80 | 0.20 | stay |
| 6 | medium | 0.80 | 0.20 | promote → hard |

Curriculum promotions: 2 (easy → medium → hard)
Detector wins 4/5 rounds consistently — environment provides genuine learning signal.

## Training

We trained Qwen2.5-3B using GRPO with HuggingFace TRL and Unsloth.

Stack:
- Model: Qwen2.5-3B-Instruct (4-bit QLoRA)
- Trainer: GRPO via HuggingFace TRL
- Efficiency: Unsloth
- Environment: HalluciNet Adversarial on HF Spaces

Training notebook: [INSERT COLAB LINK]

## Links

- **HF Space**: https://shreyshringare-hallucinet.hf.space
- **GitHub**: https://github.com/shreyshringare/hallucinet_round2
- **Colab Training**: [INSERT COLAB LINK]
- **Round 1**: https://rushikeshbathe096-hallucination-detector.hf.space

Built by Team TLE for Meta PyTorch OpenEnv Hackathon × Scaler 2026.
Abeer Nikhil Sane | Shreyas Shringare | Rushikesh Bathe | SPIT Mumbai
