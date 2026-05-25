You are fixing issues on an existing pull request for the game_players repository.

Before editing:

- Read AGENTS.md.
- Read the PR context file pointed to by `CODEX_PR_CONTEXT` if that environment variable is set.
- Inspect the current diff against the PR base branch.

Make the smallest changes that address the review feedback and CI failures. Preserve user intent, do not rewrite unrelated code, and do not add large generated artifacts.

After editing:

- Run `pytest` if practical.
- Summarize the fixes and any remaining risk in your final response.
