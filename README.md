# Transformer Models From Scratch

![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

This is my personal study repository for the Transformer architecture and some of its extensions. I'm rebuilding things from scratch.

> [!NOTE]
> **This repository is for my study purposes only.**
> It is not intended for any practical application. This repository only serves as a archive to host my code implementations for papers as I'm learning.


## What's inside

Components and variants I've been working through, roughly in the order I've tackled them:

- Scaled dot-product attention
- Multi-head attention
- Positional encoding (sinusoidal)
- Learned positional embeddings
- Layer normalization (pre-norm & post-norm)
- Rotary position embeddings (RoPE)
- KV cache for inference
- Grouped-query attention (GQA)
- Small end-to-end examples (character-level LM, tiny translation model, minimal GPT-style decoder)

## Getting started

```bash
git clone https://github.com/farzam-khodajoo/transformers-from-scratch
cd transformers-from-scratch
pip install -r requirements.txt
```

Run the tests to make sure everything is wired up:

```bash
pytest tests/
```

## Design principles

1. **Readable over fast.** No fused kernels, no clever tricks — just code I can read top to bottom.
2. **Shape-check everything.** Every module has a comment or test showing input/output shapes.
3. **Papers as source of truth.** Each component links back to the paper or reference it came from.
4. **No framework shortcuts.** If PyTorch has a built-in, I try to reimplement it first before comparing.


## License

MIT — but again, please don't use this in anything that matters.

---

Feel free to fork this if you're on a similar learning journey. Suggestions, corrections, and "you got this wrong" issues are very welcome.