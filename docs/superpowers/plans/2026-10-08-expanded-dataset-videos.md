# Expand requested dataset video coverage

Authorized scope: a1 (Genie1, ARX Lift2, Split Aloha), RoboMind v1 (AgileX, dual FR3), RoboChallenge v2 (ALOHA, DOS-W1), RoboMind v2 AgileX, ABC130k YAM. One continuous sample per embodiment, 9 additions. Read the user's Mora v0.3 audit and authoritative source metadata/code; use actual V0_2/V0_3 physical release paths, not an inferred directory name.

Investigate native TCP and matched URDF fixed flange→tool chains/config before deciding supervision is unavailable. Distinguish missing tool semantics from missing camera/base mounting or unavailable robot state. Independent FK checks need source joint labels and a separately defined comparison pose; derived EEF must never be compared against itself as GT.

Parallel independent extraction/audits: a1, RoboMind, RoboChallenge+ABC. Root owns shared FK-bundle ingestion, video/report integration and final validation. Source data and training readers/config remain unchanged.

- [x] Audit and export 9 bounded source clips with head/wrist images, TCP masks and FK provenance.
- [x] Add validated generic FK bundle ingestion and source-derived labeling; support new report entries without weakening prior sources.
- [x] Render head endpoint/relative restoration, camera projection, relative XYZ, FK and available left/right wrist views.
- [x] Integrate all clips plus explicit availability/URDF/TCP evidence table into original portable HTML/ZIP.
- [x] Verify source joins, exact frame timestamps, FK error meaning, video decodes, embedded media hashes, tests, screenshots and serving latest content.
