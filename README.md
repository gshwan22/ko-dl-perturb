# Korean dependency-distance perturbation corpora and fine-tuned model outputs

Resource accompanying *Beyond Mean Dependency Distance: Measuring the Syntactic Organization of Generated Korean as a Distribution over Typed Relations*.

## Contents

| Path | Description |
|---|---|
| `data/corpora/` | Five aligned training corpora of 30,000 Korean QA documents each: `nu0` original, `nu1` LLM rewrite toward shorter dependencies, `pi-minus` / `pi-plus` tree-fixed reorderings that decrease / increase mean dependency distance, `pi0` tree-fixed reordering with mean dependency distance held within 1% of the original |
| `data/splits/` | Validation set (`val.jsonl`) and the 1,000 evaluation prompts (`eval_prompts.jsonl`) |
| `outputs/30k/<condition>/<model>.jsonl` | 29,000 generated answers: six models x five conditions, 1,000 each (`pi0` without EXAONE) |
| `outputs/536k_pi0/` | 5,000 answers from models fine-tuned on the full 536,153-document `pi0` corpus |
| `features/stanza/`, `features/udpipe/` | Per-document parsed features (relation-specific distances and counts) for every output file and the reference |
| `features/corpus_profiles_30k.json` | Relational profiles of the five training corpora |
| `code/` | Perturbation, training, generation, parsing, and measurement scripts |

File names under `features/` keep the internal condition codes: `C0_original` = `nu0`, `C1_gpt_dlmin` = `nu1`, `C4_algo_dlneu` = `pi0`, `C2_algo_dlmin` = `pi-minus`, `C3_algo_dlmax` = `pi-plus`, and `C4b_algo_dlneu536k` = `outputs/536k_pi0`.

Each JSONL record in `data/corpora/` holds a `messages` list (system, user, assistant) and a `meta` object with the document identifier `doc_id`, the domain, and the split. Only the assistant turn is perturbed.

## Provenance

The source QA pairs were generated at KISTI with GPT-4o mini through the OpenAI API (temperature 0.8, November 2025) across twelve domains. The `nu1` rewrite was produced with GPT-4.1 (temperature 0.2); the prompt is given in Appendix B of the paper. Use of these model outputs is additionally subject to the OpenAI terms of use.

## Pipeline

1. `code/perturb/` builds `pi0`, `pi-minus`, `pi-plus` from the original with a single Stanza parse per sentence.
2. `code/train/` fine-tunes each model with LoRA on each 30,000-document corpus.
3. `code/generate/` produces answers to `data/splits/eval_prompts.jsonl` (at most 512 new tokens; decoding settings are in the script).
4. `code/parse/` parses outputs with Stanza and UDPipe; `code/measure/` computes collapse rates, relation-conditioned profiles, the appendix tables, and the figures.

Scripts reference the absolute paths of the original servers; adjust the path constants at the top of each script.

## Licenses

Data (`data/`, `outputs/`, `features/`): CC BY 4.0, see `LICENSE-DATA`. Code (`code/`): MIT, see `LICENSE`.

## Citation

See `CITATION.cff`.
