# Missing-data recovery, 2026-11-09

User approved excluding unavailable datasets and creating separate continuation runs.
Original runs and checkpoints remain unchanged. Nearby CPFS directories were searched;
no identical replacement was found. Restore optimizer and weights from steps 4000/2000.

Implementation and verification sequence:

1. Add `filtered_global_shuffle`, binding parent population, seed, NumPy runtime,
   and sorted excluded source-ID intervals into the sampling contract. Construct
   the parent permutation, delete excluded IDs, and compact surviving IDs. Test
   exact suffix preservation, every-sample coverage, cursor endpoints, and invalid
   overlapping/out-of-range intervals. Existing order contracts remain unchanged.
2. Derive a new YAML and SQLite index by removing only the 133 authorized repos.
   Preserve every surviving episode digest, source stamp, run grid and segment;
   remap repository IDs and sample starts. Validate the full new index normally.
3. Rebind a copy of the existing normalization registry to the retained selection,
   explicitly recording inherited statistics and parent registry provenance.
   Run the normal Stage2 metadata audit and offline preparation for both profiles.
4. Create new checkpoints with new identities and remapped cursors, retaining
   original update counts and optimizer tensors. Record old/new hashes and lineage;
   verify checkpoint contents and unchanged model/dependency state.
5. Commit isolated code, launch `stage12_entry.sh train --resume` on the eight
   existing DLC pods with new output/config paths and rendezvous ports. Observe
   actual optimizer updates beyond both source checkpoints and finite losses.
6. Archive conversation and operational evidence; document and validate reusable
   recovery workflow once exercised. Do not claim an unchanged original experiment.

The new order is the original order with unavailable samples removed. Already-seen
surviving samples remain before the mapped cursor; no surviving sample is replayed
or skipped by cursor migration. Batch boundaries change. Past influence of removed
datasets remains in the learned weights and optimizer, as in any continuation.
