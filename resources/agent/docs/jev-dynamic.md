# Jev Dynamic

Dynamic lets the coding agent ask Jev ad hoc, typed questions through the native
`jev_decide` tool. The agent selects the primitive and supplies the relevant state,
instructions and answer criteria.

Enable it for the current chat:

```text
/jev active
/jev feature dynamic
```

`/jev feature dynamic on` is equivalent. `/jev feature dynamic off` disables it.
Dynamic also works in Compare + Active. Compare alone does not expose the tool.

**Full Jev enables Dynamic automatically**, including for existing Full Jev
profiles. There is no extra switch to turn on. To change individual settings,
first remove the overlay with `/jev full-jev off`; this restores saved settings.

Try:

```text
Get Jev to flip a coin.
Ask Jev whether this patch addresses the error in the supplied log.
Have Jev rate these three candidate solutions from poor to excellent.
```

The agent uses:

- **Choice** for one option from a defined set.
- **Noul** for the probability that a yes/no condition holds.
- **Score** for a rating over two to ten ordered descriptions.

Independent questions can be batched in one request (up to 64). Jev returns typed
answers and probabilities; it does not generate explanations. Choice and Score
also return confidence. The tool includes the reported model, latency, attempt
count and actual token usage when supplied by the service. Usage contributes to
the current session's Jev statistics; Dynamic does not claim estimated savings.

In chat, Jev calls appear as a compact status row in overview and details mode.
Use **Ctrl+O** to reach all output and reveal the full request and result JSON;
the next press collapses them again. This follows the saved chat detail setting
for new, resumed, and attached chats. Errors keep a short visible summary when
collapsed. The complete data remains available to the coding model and in the
saved conversation.

For a random choice, the agent requests local sampling of a Choice distribution:

```json
{
  "state": {"request": "Flip a coin"},
  "questions": {
    "coin": {
      "type": "choice",
      "instructions": "Choose a side of the coin.",
      "criteria": {"heads": "Heads", "tails": "Tails"}
    }
  },
  "sample": ["coin"]
}
```

Each call draws fresh randomness from Jev's returned probabilities and preserves
the raw answer alongside the sampled result. This is not a guaranteed fair 50/50
coin toss. Exact fair randomness should use an ordinary random-number generator.

Dynamic reuses the configured Jev model and credential. It sends the supplied
state/questions to the existing Jev service, validates responses, and observes
request limits, retries, cancellation and session stop. Changing settings or
credentials during a request invalidates it. Invalid or incomplete answers produce
an error rather than a fabricated decision. These explicit calls do not trigger a
separate LLM comparison, even when automatic decisions are in Compare + Active.

An attached daemon must advertise `jev_dynamic`. Older daemons continue to attach,
but Dynamic is unavailable until the daemon is updated.
