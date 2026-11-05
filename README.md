# Measure Before You Repair

Code for the paper *Measure Before You Repair: An Explainable Agent for Blind Restoration of Multi-Artifact 3-D MRI*
by Necati Sefercioglu, Mehmet Ozan Unal, Metin Ertas, Altan Cakir and Isa Yildirim (Istanbul Technical University).

Paper: [arXiv:2611.03996](https://arxiv.org/abs/2611.03996)

Dx-Rx is an agent in which a multimodal large language model (MLLM) measures suspected artifacts before it repairs
them. It receives one 3-D T1-weighted brain volume with an unknown mix of five artifacts (noise, spike, ghosting, bias
field, motion), calls five diagnostic tools and four restoration tools in any order it chooses, checks each repair, can
return to any earlier version, and ends with a radiology-style report that is checked against its own run log.

## What is here

| Folder | Contents |
|---|---|
| `src/bamr/data/` | IXI preparation, 3-D artifact simulation, test-set assembly |
| `src/bamr/tools/` | the five diagnostic tools, *look* and *compare* |
| `src/bamr/fixes/` | the four restoration tools (DRUNet for noise) |
| `src/bamr/agent/` | Dx-Rx: loop, prompts, report check, the compared methods, OpenRouter client |
| `src/bamr/eval/` | scoring, and the guard that keeps labels and clean volumes away from the tools and the agent |
| `scripts/` | one command per step, below |
| `configs/` | `dataset_v1.yaml` (preparation settings), `dataset_v2.yaml` (artifacts, levels, test set) |
| `splits/` | the two subject lists the test set is drawn from (IXI subject ids) |
| `tests/` | unit tests |

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run python scripts/get_weights.py     # DRUNet weights (KAIR v1.0 release) → ~/.cache/bamr, sha256 checked
cp .env.example .env                     # then put your OpenRouter key in .env
```

HD-BET downloads its own weights on first use. Apple MPS is used when present, else the CPU.

## Data

The scans are not included. Download the IXI T1 scans (CC BY-SA 3.0) from
https://brain-development.org/ixi-dataset/ and place the repository next to a `dataset/` folder:

```
<any folder>/
  measure-before-you-repair/    this repository (a git clone: the build records the commit)
  dataset/
    IXI-T1/                     IXI002-Guys-0828-T1.nii.gz, …
```

Everything else under `dataset/` is written by the scripts. MR-ART (OpenNeuro ds004173, CC0) is downloaded by step 2.

## Steps

```
# 1  Prepare IXI: HD-BET brain mask, N4, 1 mm, white matter = 1
uv run python scripts/prep_ixi.py

# 2  Calibrate the levels on MR-ART (optional: the result is already in configs/dataset_v2.yaml)
uv run python scripts/mrart_check.py select
uv run python scripts/mrart_check.py download
uv run python scripts/mrart_check.py prep
uv run python scripts/calibrate_v2.py mrart
uv run python scripts/calibrate_v2.py fit
uv run python scripts/calibrate_v2.py grid

# 3  Build the test set (148 volumes) and the development volumes
uv run python scripts/build_v2.py split
uv run python scripts/build_v2.py build --allow-dirty   # first set stage3.frozen to false in configs/dataset_v2.yaml

# 4  Run every method on the test set, then score (needs .env)
zsh scripts/test5.sh compute
zsh scripts/test5.sh models
```

Step 4 is the exact recipe of the paper's test run; results go to `dataset/task4-results/stage5-test/` (`scores.json`,
`rows.csv`, one folder per method and volume with the output volume, the log and the report).

One volume with one method:

```
uv run python scripts/run_agent.py smoke --model terra --out demo     # a model runs only after its smoke test passed today
uv run python scripts/run_agent.py run --system agent --model terra --cases t0122 --out demo
uv run python scripts/score_agent.py --out demo
uv run python scripts/paper_cards.py --out demo                        # the report as a page
```

| `--system` | Method in the paper |
|---|---|
| `chain_u` | Apply-all pipeline |
| `chain` | Rule-based pipeline |
| (`score_agent.py --oracle`) | Oracle pipeline |
| `router` | Single pass, no diagnostic tools |
| `router_readings` | Single pass + diagnostic tools |
| `agent_eyes` | Loop, no diagnostic tools |
| `agent` | Dx-Rx |

| `--model` | MLLM (through OpenRouter, provider pinned) |
|---|---|
| `haiku` | Claude Haiku 5.5 |
| `sonnet` | Claude Sonnet 5.5 |
| `luna` | GPT-6 Luna |
| `terra` | GPT-6.1 Terra |
| `qwen` | Qwen3.8-27B |

The tools' thresholds and scales were set on development volumes with `scripts/score_tools.py` and
`scripts/score_fixes.py`; the values are in the code.

## Tests

```
uv run pytest
```

Tests that need the data skip themselves when it is absent.

## Citation

```bibtex
@misc{sefercioglu2026measurerepairexplainableagent,
      title={Measure Before You Repair: An Explainable Agent for Blind Restoration of Multi-Artifact 3-D MRI},
      author={Necati Sefercioglu and Mehmet Ozan Unal and Metin Ertas and Altan Cakir and Isa Yildirim},
      year={2026},
      eprint={2611.03996},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2611.03996},
}
```

## License

MIT, see `LICENSE`. `src/bamr/fixes/drunet.py` re-implements the DRUNet network of KAIR (MIT); see
`THIRD_PARTY_NOTICES.md`.
