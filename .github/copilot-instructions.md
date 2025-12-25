## Purpose

This file gives concise, repo-specific guidance for AI coding agents (e.g., GitHub Copilot) to be immediately productive in this NaturalDreamer-agent codebase.

## Overview (big picture)

- High-level: This repo implements a readable DreamerV3-style agent. The main training loop is in [main.py](main.py) which instantiates `Dreamer` (in [dreamer.py](dreamer.py)).
- World model components (encoder/decoder, prior/posterior, recurrent network, reward/continue predictors) live in [networks.py](networks.py). The `Dreamer` class composes these.
- Experience storage: replay logic is in [buffer.py](buffer.py) (note: borrowed from SimpleDreamer; author mentions it should be remade).
- Environment wrappers and small gym helpers are in [envs.py](envs.py); config loading, seeding, plotting and util helpers are in [utils.py](utils.py).

## Key files to reference when making changes

- [main.py](main.py): CLI entrypoint. Default config: `car-racing-v3.yml`. Use `--config` to change.
- [dreamer.py](dreamer.py): `Dreamer` class — contains `worldModelTraining`, `behaviorTraining`, `environmentInteraction`, checkpointing, and buffer usage.
- [networks.py](networks.py): All networks (EncoderConv, DecoderConv, PriorNet, PosteriorNet, Actor/DiscreteActor, Critic, etc.).
- [buffer.py](buffer.py): `ReplayBuffer` interface and quirks (stores discrete actions as int64; continuous as float32).
- [envs.py](envs.py): `GymPixelsProcessingWrapper` (transposes and normalizes image obs to [C,H,W]) and `CleanGymWrapper` (compat layer for gym's terminated/truncated to done).
- [utils.py](utils.py): `loadConfig()` uses `findFile()` so config yml can be located in subfolders; `seedEverything()` and plotting utilities live here.

## Project workflows and examples

- Install deps: `pip install -r requirements.txt` (see `requirements.txt`).
- Train (default):

  ```bash
  python main.py --config car-racing-v3.yml
  ```

- Resume from checkpoint: set `resume: true` and `checkpointToLoad` in the yaml config (main builds a `runName` and expects `.pth` under configured checkpoint folders).
- Quick discrete-action sanity test: run `python test_discrete.py` to validate `getEnvProperties()` and wrappers for ALE environments.

## Conventions and important patterns (concrete)

- Configs are AttrDict-like objects returned by `loadConfig()` (see `utils.py`): access is `config.section.param`.
- Observation layout: wrappers transform images to shape (C, H, W) and normalize to [0,1]. When editing encoders/decoders, expect that shape.
- Discrete vs continuous actions: code checks `isDiscrete` from `getEnvProperties()` and:
  - Converts discrete actions to one-hot before passing to the recurrent model.
  - `ReplayBuffer` stores discrete actions as int64; sampling returns torch.long for discrete actions.
  - Use `DiscreteActor` in `networks.py` for discrete action support.
- World model training loop (in `dreamer.worldModelTraining`) uses OneHotCategoricalStraightThrough for discrete latents. Be cautious when changing latent distributions.
- Checkpoint semantics: `Dreamer.saveCheckpoint()` stores model and optimizer state dicts plus counters. `loadCheckpoint()` expects the same keys.

## Testing & debugging tips

- If GPU is unavailable, code falls back to CPU (`torch.device` selection in `main.py`). For determinism call `seedEverything()` (already used by `main`).
- To debug shapes, insert quick asserts near encoders/decoders and run a single `dreamer.environmentInteraction()` episode (it's safe and exercises buffer / encoder path).
- Metric/logging: CSVs are appended by `saveLossesToCSV()` and plotted by `plotMetrics()` (HTML output). Use these to inspect training traces.

## Known caveats (from author notes / README)

- Primary tested environment: `CarRacing-v3` (continuous image-based). The code was developed on Linux and Windows support is untested.
- The buffer is a direct import from SimpleDreamer and the author notes it should be remade — be conservative when refactoring it.
- TwoHot loss and some discrete features are incomplete or experimental; follow existing `rewardPredictor` (Normal) code paths as default.

## Editing guidance for agents

- Make minimal, focused edits. Example: to add a new encoder block, edit `EncoderConv` in [networks.py](networks.py) and adapt `DecoderConv` reconstruction size accordingly.
- When adding CLI flags, prefer extending the YAML config and reading via `loadConfig()` (and avoid hardcoding paths). `findFile()` lets configs live in subfolders.
- When changing training loops, preserve counter updates (`totalEnvSteps`, `totalGradientSteps`, `totalEpisodes`) and checkpointing behavior so resuming works.

If anything's unclear or you want additional examples (config keys, sample training run logs, or targeted unit tests), tell me which area to expand.
