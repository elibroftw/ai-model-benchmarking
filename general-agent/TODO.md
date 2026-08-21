# Roadmap

## Skill: authoring skills for tasks that have none

The skill system only helps when a relevant skill already exists. Today, a
task with no matching skill gets the agent's default behavior — which is
exactly where it wastes effort on incidentals (see
[`scribbling-on-images`](skills/scribbling-on-images.md), written only after
watching models burn turns matching fonts instead of writing legible digits).

Build a bootstrapping skill — call it `writing-skills` — that the agent loads
when it notices no existing skill covers what it is doing. It should tell the
agent how to:

- **Recognize the gap.** What signals that the current sub-task deserves a
  skill? Repeated work across sessions, a decision it keeps re-litigating,
  or effort spent on something incidental to the actual goal.
- **Find the best practices.** Where to look, how to tell a real convention
  from one blog post's opinion, and how to weigh conflicting advice. This is
  the hard part and the reason the skill is worth writing: "search the web"
  is not a method.
- **Decide what matters vs. what doesn't.** The most valuable line in a skill
  is usually the prohibition — what NOT to spend effort on. A skill that only
  lists best practices, without saying which details are noise, doesn't solve
  the problem the skill system exists for.
- **Write it in the house format.** Frontmatter `name` + `description`, where
  the description is a *when-to-use trigger* rather than a summary, since
  that line is what the agent matches against to decide whether to load it.
- **Include an escape clause.** Every skill needs to say when it does not
  apply, or it will be followed in cases where it is wrong.

### Open questions

- Should the agent write skills mid-task (interrupting the work) or propose
  them at the end? Mid-task risks derailing; end-of-task risks losing the
  context that made the gap visible.
- Do generated skills land in `skills/` directly, or in a staging area for
  human review before they start steering future runs? A bad auto-generated
  skill is worse than no skill, because it is trusted.
- How do we keep skills from accumulating into a pile that contradicts
  itself? Some notion of review, merging, or expiry is probably needed.
