# Design notes

The decisions behind the pipeline, and the places where a number is still a
guess. Moved out of the README on 2026-10-04 to keep that page short; nothing
here is duplicated elsewhere in `doc/`.

**A newsletter is a curated list, and that is the best signal here.** An editor
chose ten items out of the day's hundreds *and* chose which one opens, so position
in the newsletter is two judgements rather than one — and it is free, and available
in every language. It is the highest-weighted term in the ranking formula.

**Tracking links are not a cosmetic problem.** Newsletter links go through a click
tracker, and until one is resolved back to `elpais.com/deportes/...` three things
are broken: attribution points at a URL that expires, the same article carries a
different opaque token in every newsletter so deduplication cannot see it is one
article, and every section blocklist silently matches nothing.

**Filtering has to ask to drop something.** An unclassified item is kept, a failed
batch is kept, an exhausted quota keeps everything left, and an item is only
dropped on its topics when *every* topic is excluded. The worst case is a noisy
digest, never an empty one — a wrongly dropped story is invisible in the output,
whereas a wrongly kept one merely ranks low.

**Configuration is not code.** Sources, filters, the ranking formula and digest
size live in three YAML files, and the prompts live in `prompts/`. Deleting a line
from `ranking.terms` removes that signal from the maths entirely, and a misspelled
term is a startup error rather than a silent no-op. `news-digest explain
<story-id>` prints the per-term breakdown, and the numbers provably sum to the
score used for ranking.

**Where sources disagree, the digest says so.** Disagreements are a separate field
on the story and a separate section in the output, never folded into the summary.
Silently resolving a factual conflict is the one thing a digest of ten outlets
must not do.

**The ranking constants are not yet calibrated.** They are carried over from the
RSS-based version of this project, where each was measured against a published top
ten — against a source list that no longer exists. Ten newsletters cannot exceed
ten publishers, so `corroboration_saturation` in particular is a guess until a real
week has been measured. Every such number says so where it is defined.

**Re-running is free and safe.** A message is fetched once, parsed once, and its
articles classified and enriched once, keyed on content hash. The mailbox is opened
read-only, so a run can neither delete a newsletter nor mark one read.
