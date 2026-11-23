# Jobs recipes

Recipes that train or reproduce open models on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs).
Each recipe is one folder with:

- `README.md`: what it does, the commands, the cost and the results of a tested run;
- `AGENTS.md`: the procedure, the checks and the stop rules for a coding agent that runs the recipe for a user;
- the scripts. They install the upstream code at pinned commits and keep their state in a
  [Hugging Face bucket](https://huggingface.co/docs/hub/storage-buckets), so a stopped Job can resume.

| recipe | what you get | GPU, time, cost |
|---|---|---|
| [tiny-voca](tiny-voca/) | a 4.5M-parameter English TTS model with two voices, trained from scratch; scores match the released checkpoint | 1 × RTX PRO 6000, about 5 h 20 min, about $15 |

## Requirements

- A Hugging Face account with Jobs access.
- The `hf` CLI (`pip install -U huggingface_hub`), logged in with a token that can write.
- A bucket: `hf buckets create <you>/<bucket> --private`.

Run the commands from the root of this repository: each command mounts its recipe folder into the Job with
`-v ./<recipe>:/scripts:ro`.
