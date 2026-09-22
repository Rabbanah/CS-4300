# U0-HW-04 Starter: Repairing a Conference Schedule

Requires Python 3.10+ and uses only the standard library.

Complete `constraints.py`, `MinConflictsAgent`, `LLMRepairAgent`, and your tests
in `test_student.py`. Do not change the supplied model, scenarios, interfaces,
metric meanings, or one-talk `move` operation.

```sh
python3 -m unittest -v test_conference
python3 -m unittest -v
python3 run_experiment.py --all --seeds 0,1,2
python3 run_experiment.py --agent minconflicts --scenario conference-1 --trace
```

No ordinary command contacts a network. Before a live run, install Tailscale and
connect to the Utah Tech Tailnet:

- **macOS:** In the pop-up used to add an account, enter
  `https://tailscale.utahtech.dev`.
- **Windows:** Open PowerShell or Command Prompt and run
  `tailscale up --login-server=https://tailscale.utahtech.dev`. Do not run this
  command inside WSL or Ubuntu.
- **Linux:** Run
  `tailscale up --login-server=https://tailscale.utahtech.dev`.

After Tailscale is connected, set the required API key to a non-secret
placeholder. The course endpoint ignores its value. Use the command for your
shell:

```sh
# macOS or Linux
export OPENAI_API_KEY=course-placeholder

# Windows PowerShell
$env:OPENAI_API_KEY = "course-placeholder"

# Windows Command Prompt
set OPENAI_API_KEY=course-placeholder
```

Then run the comparison. Windows users can enter the Python command on one line
instead of using the backslash continuations shown here:

```sh
python3 run_experiment.py --compare --live \
  --model Gemma-4-26B-A4B-it-oQ4e-mtp \
  --endpoint http://golem:8000/v1 --temperature 0 --seeds 0,1,2
```

`--compare` runs min-conflicts and the LLM consecutively for each explicit
scenario/seed pair from the same immutable start and budget. `--all` always
remains offline. Do not place any real credential in `OPENAI_API_KEY`, source
files, reports, shell scripts, or commits. The `ScriptedClient` is for
deterministic tests, not empirical LLM evidence.

The course service currently provides these exact model identifiers:

- `Gemma-4-26B-A4B-it-oQ4e-mtp`
- `Qwen3.8-27B-oQ4e-mtp`
- `Qwen3.8-Flash-Next-oQ4e-mtp`

`LLMRepairAgent` must permit `history_limit=0` (send no prior outcomes), reject
negative limits, reject duplicate JSON object keys, and continue after ordinary
client failures. Count every attempted call and record only the exception type,
not potentially sensitive exception text.
