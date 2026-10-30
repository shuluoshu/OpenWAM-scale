# Remove direct Stage1 readout

User explicitly requested deleting the direct readout branch. Keep the query encoder, joint image/TCP flow decoder, constant observation latent prefix and existing condition projection unchanged.

1. Add failing contracts: no head/weights/loss; joint-flow gradients reach image/TCP queries; strict legacy bundle migration.
2. Remove GroupReadout and its losses, constructor/config/logging fields; retain masked joint flow supervision.
3. Model-only legacy load drops readout weights with warning. Full optimizer resume refuses obsolete topology. Stage2 encoder-only load is unchanged.
4. Update operator architecture and distinguish local Wan decoder compatibility from PatchWAM FLUX fusion.
5. Run query, history, Stage2, bundle and finite-epoch trainer regression tests. Preserve all live pinned snapshots and jobs.
