# What "splicing" means here (figure source)

Scratch doc: the ASCII explanation from chat, saved verbatim as source material
for a proper figure later. Not part of the technical note itself.

"Splice" is the film/tape-editing word: physically cutting two separate reels
of tape and joining them so playback continues seamlessly from one recording
into a totally different one, with no audible seam. That's what happens here,
to a KV cache instead of magnetic tape.

## Step 1: two separate, unrelated recordings

```
REAL conversation (system prompt + user question)
never touches the shadow text at all

  ┌──────────────────────────────┐
  │ [sys constraint][user Q]     │  ── forward pass ──►  cache A
  └──────────────────────────────┘   positions 0..29     (30 slots of
                                                            K,V per layer)

SHADOW text (forbidden or neutral topic)
processed completely separately, model has no idea cache A exists

  ┌─────────────────────────────────────────────┐
  │ "A patient with X is showing signs of Y..."  │  ── forward pass ──►  cache B
  └─────────────────────────────────────────────┘   positions 30..509    (480 slots)
                                                      ↑
                                    position_ids are set to START at 30,
                                    not 0 — so this "tape" is pre-labeled
                                    to slot in right after cache A, even
                                    though it was recorded independently
```

## Step 2: the splice itself

```
  cache A (real, positions 0-29)      cache B, first `num_append` slots taken
  ┌──────────────────────┐            ┌───────────────────────┐
  │  real tokens          │  torch.cat │  phantom tokens        │
  │  (untouched)          │ ────────► │  (never said, never    │
  │                       │           │   generated)            │
  └──────────────────────┘            └───────────────────────┘
              └───────────────┬────────────────────┘
                    spliced cache, positions 0..(29+num_append)
                    generation resumes from the end of THIS

The model has no way to tell "phantom" from "real" — they're just
valid K,V vectors sitting at valid positions. That's the whole trick:
nothing marks the seam.
```

## The "dose" = how much of the shadow tape gets spliced on

Shadow cache B has 480 slots total (480 tokens of shadow text). Dose = what
fraction of it actually gets grafted onto cache A:

```
480 total shadow-cache positions:
|■■■■■■■■■■■■■■■■■■■■■|································|
0                    120                              480
 └─── 25% dose ──────┘        (remaining 75% never used)

|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■|········|
0                                                    360      480
 └───────────────── 75% dose ─────────────────────┘  (unused)

|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■|
0                                                             480
 └───────────────────── 100% dose: whole tape spliced on ────┘
```

## ragged vs clean cut = *where exactly* the tape gets cut

75% of 480 tokens lands at position 360, which is usually mid-sentence:

```
shadow text:  "...experiencing a flare-up of their psoriatic arthritis. A patient wi—"
                                                                         ↑
                                              raw 75% token count lands HERE

RAGGED cut — splice stops at the raw count, mid-word:
  "...psoriatic arthritis. A patient wi ✂"
                                        └─ splice ends here, mid-word

CLEAN cut — snapped backward to the nearest full-sentence ending:
  "...psoriatic arthritis. ✂ A patient with..."
                          └─ splice ends here, clean sentence boundary
```

So "cache injection at ragged75" = take the shadow text's KV cache, keep
only the first 75% of it (cut wherever that raw token count lands), splice
that chunk onto the end of the real conversation's cache, and let the model
keep generating from there — attending back over a sequence that is
genuinely, tensor-for-tensor, `[real conversation][phantom tape]`.

## The one token that gets a genuine joint forward pass

The spliced cache only holds `num_append - 1` phantom positions (deliberately
one short). `extended_input_ids` passed to `generate()` has the full
`num_append` tokens, so `generate()` runs a real forward pass for exactly
one token: the last one. Every other phantom position's K,V was computed
shadow-text-only, with zero awareness the real prompt exists. This one
token is the only place "real prompt" and "phantom content" are honestly
attended over together.

```
  cache A (real, 0..29)     cache B, FIRST (num_append-1) positions      1 fresh token
  ┌────────────┐            ┌─────────────────────────────┐            ┌──────────┐
  │ real tokens │  torch.cat │ phantom K,V (computed        │  generate()│ held-back │
  │             │ ────────► │ shadow-only, no awareness     │  forward   │ token,    │
  │             │           │ real prompt exists)           │  pass ───► │ genuinely │
  └────────────┘            └─────────────────────────────┘            │ joint     │
                                                                         └──────────┘
                                                                              │
                                                          this token's logits pick
                                                          the model's actual first
                                                          generated token
```

Concrete examples, Qwen tokenizer:
- medical clean75 (num_append=360): held-back token is `'.'`, the period
  ending `"...restore blood flow to the heart muscle."` — clean cuts are
  *defined* as landing on sentence-ending punctuation, so this is
  essentially guaranteed.
- finance ragged75 (num_append=341): held-back token is `' at'`, a
  mid-clause fragment from `"...COST) is rated a hold at a price target of
  $..."` — ragged cuts land wherever the raw count lands, no regard for
  clause boundaries.

## Correction / precision note on the ragged→clean conversion

Clean cut is **not** built by appending anything, and specifically does
**not** involve adding an `<eos>` token. It's the reverse operation:
`_snap_to_clean_sentence_boundary` (experiment.py) walks the ragged target
count *backward* to the nearest sentence-ending punctuation (`.`, `!`, `?`)
that was already present in the original shadow text, via
`_sentence_boundary_token_counts`, which tokenizes the shadow text truncated
after each of its own naturally occurring sentences and records the
cumulative token count at each of those points. Clean cut then picks the
largest such count that's `<= ragged_target`. So clean is always a
same-length-or-shorter splice than ragged, made of tokens that were already
going to be in the shadow text — nothing generated, nothing inserted, no
special tokens added. The one edge case: if the shadow text's very first
sentence alone already exceeds the ragged target, clean falls back to
including that whole first sentence anyway (a zero-length "clean cut"
wouldn't mean anything), so in that one case clean can end up very slightly
longer than the ragged target — still without adding any token, just by
keeping one whole naturally-occurring sentence intact.
