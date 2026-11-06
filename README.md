# N/Agency++

**Auditing the Public Surface of User-Facing LLM Agents**

[Project page](docs/index.html) | [Experiment code](research/)

N/Agency++ separates model proposals, permission decisions, and rendered public
fields through Permissioned Action Generation (PAG). This release contains the
experiment code, synthetic benchmark, saved results, and a minimal project page.
It does not claim conference acceptance or general semantic safety.

## Evaluation Scope

- Historical V1: five comparison conditions on five backbones. PAG V1 directly
  reads gold permission and is an oracle diagnostic, not a deployable policy.
- Live V2/V3: 9,000 saved generations, each replayed under three display policies.
  The 27,000 audit views are not additional model generations.
- V2: final-line exposure is 1.00%; full-surface exposure is 1.23%.
- V3: the measured exposure rate is 0.00%, but the controller shares the audit's
  annotation-conditioned detector (`gold_permission` and category). This is
  detector-relative containment, not an independently validated safety result.
- On 6,000 denied generations, V3 retains 21.00% detector-defined non-refusal and 65.80% visible
  non-withheld intent. These denominators differ from all-generation rates.

The original Godot service and model weights are not bundled. Experiment
logic and existing result values are preserved; no new model inference was
used to prepare this release.

## Run Without A GPU

Use Python 3.10 or later from the repository root.

```sh
python -m pip install -r requirements.txt
python research/validate_pag_benchmark.py
python research/emnlp_industry_strengthening_analysis.py
python research/naacl_deep_analysis.py
python research/naacl_latency_budget.py
python research/naacl_component_ablation.py
# These two commands require the local-only inputs described below.
python research/naacl_family_ablation.py
python research/naacl_postprocessing_benchmark.py
python -m unittest discover -s research -p 'test_naacl_*.py' -v
```

The analysis commands replay saved outputs and regenerate the
scoped tables in `results/`. They do not call a model. The benchmark has 600
synthetic prompts in six rulebook categories; see
[the rulebook](docs/research/pag-policy-rulebook.md).

**Local-only inputs:** the five `pag_eval_llamacpp_*_schema_policy_full.csv`
files for the reported backbones and `pag_runtime_v3_public_surface_audit.csv`
are not distributed in this public repository. Family ablation and CPU replay
require those six locally held files in addition to the published inputs.
Their filenames/hashes, analysis code, derived summaries, and all measured
CPU timing blocks are public; these two analyses cannot be rerun from the
public checkout alone. The other saved-runtime analyses use published inputs.

## Additional Diagnostics

The paper adds nine analyses without additional generations:

- **Whole-request audit pass (Table 1b):** a nonempty final line with no surface,
  raw-output, denied non-refusal, denied-intent, or allowed-refusal failure.
  Paired V2 full-surface/V3 pass rates are 2,991/9,000 (33.23%) and 5,043/9,000
  (56.03%), a gain of 22.80 percentage points (95% prompt-bootstrap interval
  [21.30, 24.27]). There are 2,052 newly passing
  observations and no newly failing observations. All 3,000 allowed final lines
  are unchanged, including nine pre-existing refusals that still fail the audit.
  Exact line preservation is not task utility; this joint pass is not the
  controller's ALLOW rate or an independently validated semantic-safety score.
- **Empirical rendering distribution (Figure 3a):** observed proposed actions
  flow to 4,146 retained reflect, 2,802 retained withhold, and 2,052 controlled
  fallback outputs (46.07%, 31.13%, 22.80%). All 1,743 unveil and 18 invalid
  proposals enter fallback. Relative to V2, only 93 final lines change;
  1,941 observations change intent only and 18 change intent/raw visibility.
  Overall final-line preservation is 98.97%. These are empirical benchmark-
  mixture transitions, not fixed-input stochastic samples or token likelihoods.
- **Paired V2-V3 generation-budget yield (Figure 3b):** joint audit passes
  arriving within a recorded generation-time budget, divided by all 9,000
  observations per policy. V2/V3 yield is 21.49/34.26% at 1.5 seconds and
  32.60/53.21% at 2 seconds. The paired gain at 2 s is 20.61 pp [19.22, 21.98]
  (95% prompt-bootstrap interval). The policies use identical generated outputs
  and generation times: this measures renderer-policy improvement, not an
  independent end-to-end speedup or superiority over oracle V1. Backbone
  curves and contrasts remain in the appendix for deployment selection.
  This is not confidence or a production SLO; generation round trips exclude
  rendering, and audit pass is not task utility.
- **Residual heterogeneity and conditional visibility (Figure 4):** the
  non-refusal matrix separates backbone/category interactions. The forest
  plot measures visible intent among detector-refused denied observations:
  2,688/4,740 (56.71%, 95% prompt-bootstrap interval [53.95, 59.42]); backbone
  rates range from 46.13% to 82.44%. Refusal is not metadata withholding.
  The full unconditional intent matrix is retained in the appendix.
- **Ablation study (Table 3):** a one-column checkmark table for Safety prompt, Schema only, Self-policy,
  Schema + V2, and Schema + V3 (ours), uniformly audited on stored condition-sweep
  outputs. Each row has 9,000 observations; the three Schema rows use identical
  proposals. V2/V3 surface flags are 0.87/0.00%, joint pass 32.03/90.37%, and
  both preserve 96.70% of allowed lines. Text-only Safety prompt has no intent
  field. The common surface auditor includes generic and annotation-conditioned
  checks, so its rates differ from the original final-only scores in panel (a).
  The generation-family rows vary prompts/contracts; only the Schema contrast
  isolates the controller. Table 2(a,b) is unchanged; the original field-attribution
  table follows directly below as Table 4. Full residual rates are retained in
  the appendix rather than squeezed into the checkmark table.
  Recorded generation latency appears beside CPU time in Table 3: 2.095 s for Safety,
  2.075 s for Self-policy, and 1.398 s for all three Schema rows. The repeated
  Schema times reflect shared outputs, not measured zero processing overhead.
- **CPU postprocessing (Table 3; appendix protocol):** five warmups and 30
  counterbalanced passes over 9,000 outputs per variant on an Intel Core
  i5-7267U, Darwin 22.6.0, Python 3.12.14. Safety/Schema/Self-policy/V2/V3
  take 3.1/7.4/8.3/16.1/11.1 microseconds per observation on average;
  SDs across pass means are 2.3/3.7/6.7/8.9/4.8 microseconds, not request-tail
  percentiles. The timed path parses outputs, applies policy, and constructs
  public fields. It excludes model generation, network, evaluation-only
  scoring, file I/O, and GUI painting. All 45,000 views match the original
  replay. V2 retains its separate policy parser; the measurements characterize
  these Python implementations, not inherent algorithmic speed superiority.
  CPU and GPU-host times are not summed into an end-to-end estimate. Per-pass
  and per-backbone results, platform details, and input/code hashes are released.
- **Renderer component ablation (appendix):** all eight combinations of G (V3
  annotation-conditioned field gate), M (denied-intent redaction), and S
  (raw/error protection), with exact V2/all-off and V3/all-on endpoints.
  Removing G, M, or S restores 1.03%, 0.03%, or 0.20% surface flags respectively.
  G alone removes final-line flags but leaves joint pass at 33.23% because
  metadata remains visible. All variants preserve the 3,000 allowed lines.
  These are paired fixed-output ablations, not new independently generated runs.
- **Audit blind spots (Table 4):** 21 of 111 V2 flags (18.9%) lie outside the
  final line: 18 raw-parse-only cases in Mistral and 3 intent-only cases in Llama.
  The paired surface-minus-final gap is 0.23 percentage points overall
  (95% interval [0.07, 0.40]); Mistral's gap is 1.00 [0.33, 1.83].
- **Context availability (Figure 5):** blanking permission/category only for
  the controller restores 90 final-line flags (1.00%), while raw suppression
  remains intact. An explicit deny-on-missing fallback retains zero measured
  exposure at the cost of 100% allowed over-refusal under complete omission.
  Pass coverage changes from 77.2% to 78.2% with the unchanged guard, and to
  zero with deny-on-missing. Coverage is the controller pass rate, not accuracy.
  The operational trade-off panel shows why joint pass needs a preservation
  constraint: complete deny-on-missing attains 66.67% joint pass by passing all
  denied cases, but retains none of the 3,000 allowed lines.

Evaluation annotations remain fixed throughout. Omission masks are
category-stratified, nested, and shared by prompt across all models/runs.
The 5th--95th percentile bands describe 100 omission masks, not population
confidence intervals. Of 3,000 backbone-prompt groups, 2,999 have identical
raw outputs across three runs; repetitions are not independent evidence.
Separately, 2,000 category-stratified bootstrap replicates resample unique
prompts with replacement while keeping every backbone and run together.
Their 95% percentile intervals describe benchmark-resampling stability, not
open-world risk. Zero-event intervals are omitted rather than interpreted as
proof of zero risk.
`results/pag_naacl_analysis_manifest.json` records the input/code hashes,
mask protocol, and output files. These tests measure the frozen detector's
sensitivity, not annotation-free semantic safety or new generation latency.

## Fresh Model Runs

Start a compatible local llama.cpp server with a separately obtained GGUF
model, then inspect the supported options before running:

```sh
python research/run_runtime_v3_live.py --help
python research/eval_pag.py --help
```

`run_runtime_v3_live.py` generates fresh outputs, so running it incurs inference
cost. The saved paper runs used an NVIDIA RTX 4060 Laptop GPU and specific
llama.cpp settings; other hardware and decoder settings need not reproduce
their latency or exact outputs. All model licenses remain applicable.

## Files

| Path | Contents |
| --- | --- |
| `research/eval_pag.py` | Historical conditions and oracle PAG V1 |
| `research/reanalyze_runtime_v2.py` | V2 output controller |
| `research/audit_runtime_v3.py` | Three public-surface audit views and V3 controller |
| `research/run_runtime_v3_live.py` | Live generation through a local llama.cpp server |
| `research/emnlp_industry_strengthening_analysis.py` | Saved-output analyses and paper tables |
| `research/naacl_deep_analysis.py` | Joint audit pass, line preservation, residuals, field attribution, context-omission replay |
| `research/naacl_latency_budget.py` | Paired policy and backbone generation-budget yield with prompt-bootstrap intervals |
| `research/naacl_component_ablation.py` | Eight fixed-output G/M/S component combinations with exact V2/V3 endpoints |
| `research/naacl_family_ablation.py` | Generation/policy families and matched Schema + V2/V3 ablation |
| `research/test_naacl_deep_analysis.py` | Controller/evaluator separation and mask-protocol tests |
| `research/test_naacl_latency_budget.py`, `research/test_naacl_component_ablation.py` | Budget boundary, component endpoint, and field-isolation tests |
| `research/test_naacl_family_ablation.py` | Native-policy semantics and fixed-proposal family controls |
| `data/` | Synthetic benchmark and subsets |
| `results/` | Historical main-sweep results, live raw/audit rows, and scoped summaries |
| `docs/` | Static project page served by GitHub Pages |

The page opens directly from `docs/index.html`; there is no frontend build or
third-party analytics. Its icons use Lucide under the ISC license (included
alongside the bundled icon library).
