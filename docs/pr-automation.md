# Pull request automation

This repository has repo-side wiring for CI, Codex review gating/autofix, Slack traces, and Linear traces. Native Codex review is configured outside git.

## What is configured in this repo

- `.github/workflows/ci.yml` installs and tests the `2048` package from the repository root on pushes and pull requests.
- `.github/workflows/codex-pr-review.yml` runs `openai/codex-action@v1` on every non-draft PR, posts Codex feedback as a PR comment for the exact PR head, and marks the required `Codex review gate` status according to whether Codex found high-priority blocking issues.
- `.github/workflows/codex-pr-autofix.yml` lets trusted users comment `/codex fix` and supports manual workflow runs for applying review feedback.
- `.github/workflows/pr-traces.yml` posts pull request lifecycle traces to Slack and Linear when the required secrets are present.
- `AGENTS.md` tells Codex how to install, test, and review this repository.

## Required GitHub secrets

Add these in GitHub under Settings > Secrets and variables > Actions:

- `OPENAI_API_KEY`: required for the Codex GitHub Action workflows.
- `SLACK_WEBHOOK_URL`: optional incoming webhook URL for PR trace messages.
- `LINEAR_API_KEY`: optional Linear API key used to comment on matching Linear issues.

Optional repository variables:

- `CODEX_AUTOFIX_ON_REVIEW=true`: also lets `.github/workflows/codex-pr-autofix.yml` attempt a fix after PR review submissions. `/codex fix` and manual workflow dispatch do not require this variable.

## Native Codex setup

For the native GitHub PR review experience:

1. Set up Codex cloud for `latioswang/game_players`.
2. Open Codex settings and turn on Code review for this repository.
3. Turn on Automatic reviews if you want Codex to review every PR without a `@codex review` comment.
4. Use `@codex review` on a PR to request a one-off review.
5. Use `@codex fix the P1 issue` when you want the native Codex cloud task to push a fix back to the PR branch.

Codex reads this repo's `AGENTS.md`, including the `Review guidelines` section.

## Slack setup

There are two useful Slack paths:

1. Install the official Codex Slack app from Codex settings, add `@Codex` to the desired channel, and mention it to start cloud tasks from Slack threads.
2. Create a Slack incoming webhook and save it as `SLACK_WEBHOOK_URL` so `PR traces` posts concise PR lifecycle messages.

## Linear setup

There are two useful Linear paths:

1. Install the official Codex for Linear integration from Codex settings. Then assign issues to Codex or mention `@Codex` in an issue comment to start cloud tasks that leave progress updates on the issue.
2. Save a Linear API key as `LINEAR_API_KEY`. The `PR traces` workflow looks for issue identifiers such as `GP-123` in the PR title, body, or branch name, then comments on the matching Linear issue.

For reliable Linear tracing, include the issue ID in the branch name or PR title, for example `GP-123-train-eval-fix`.

## Recommended branch protection

Protect `main` with:

- Required status check: `Python tests`.
- Required status check: `Codex review gate`.
- Require branches to be up to date before merge.
- Allow auto-merge only after CI and the Codex review gate pass.

Keep fully automatic merge disabled unless you explicitly want Codex to merge after checks pass.
