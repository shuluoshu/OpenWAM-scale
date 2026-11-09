# Fully causal history and H100 switch

User authorizes causal history, repository submission, and switching the H100
experiment to this latest code with timing/progress reporting. Preserve the
other experiment's observed state. Continue in the existing isolated worktree.

## Causal contract

Retain the 32-token history budget and 120 × 3072 C* interface. Assign each
history query a chronological frame cutoff spanning the ten input slots. Its
pooling attention may read only valid frames at/before that cutoff. Keep the
real-age PE. Prefixes with no valid frames produce zero, masked tokens.

In native fusion, a history query may read only valid history tokens with no
later cutoff and may never read current image tokens. Current tokens may read
all valid past tokens and the same-time current spatial tokens. The fixed
instruction remains available as conditioning. No future target enters C*.

Introduce explicit `causal_history` mode and v3 bundle/config contracts, retaining
v1/v2 behavior. Test perturbations of later historical frames and current images
against earlier pooled/adapted history tokens, along with masks, empty history,
language gradients, bundle/runtime compatibility and old-mode regressions.

## Execution

1. Add and observe failing causal tests, then implement pooling/fusion masks,
   metadata/runtime contracts and separate causal text presets.
2. Run focused CPU/native CUDA verification and independent review. Commit and
   push the branch to the configured user fork.
3. Read-only inspect H100 liveness, exact checkpoint/model/optimizer state and
   launch control. Prepare a separate pinned snapshot and output/config paths.
   Preserve durable state and explicitly migrate compatible weights/state;
   never relabel an incompatible old optimizer checkpoint as an ordinary resume.
4. Switch only H100 after the prepared version passes validation. Re-query all
   nodes/processes, observe new finite-loss training updates, report actual
   inherited/new steps, cursor, wall time, loading overhead and steady update
   timing when measurable. Archive the full visible dialogue and evidence.
