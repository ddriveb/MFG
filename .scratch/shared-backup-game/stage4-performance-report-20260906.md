# Stage 4 performance execution probe — 2026-09-06

This is a timing gate report only. It is not Fit, Validation, a candidate
artifact, or an experiment result.

## Frozen probe

- namespace: `shared-backup-game:v1:stage4-fit`
- macro seed: `20260905`
- common path: `0`
- population: `0`
- population size: `N=8`
- realized Tokens: `1,276`
- `c_B=0.5`, degraded slowdown `2.0`, hedge delay `1.5`
- logical processors reported: `12`
- warm-up: one run per mode/rule
- measured repetitions: five per mode/rule

The reference call used `simulate_shared_backup`. Optimized full used the raw
trace entry path, prepared full reused one immutable `PreparedTrace`, and
prepared tagged retained Expert `0` while evolving all eight Experts.

## Per-call timings (seconds)

Values are `median / max`; all five raw observations were retained in the
execution log of this session.

| Rule | Reference full | Optimized full, raw | Prepared full | Prepared tagged |
|---|---:|---:|---:|---:|
| NNNN | 0.5012425000022631 / 0.5116986999928486 | 0.3755918999959249 / 0.3849624000140466 | 0.36952649999875575 / 0.3699672000075225 | 0.2800223000231199 / 0.2834067000076175 |
| NSSN | 0.4922091999906115 / 0.49794630001997575 | 0.3897220000217203 / 0.39355609999620356 | 0.37962160000461154 / 0.3833575999888126 | 0.29547519999323413 / 0.29787959999521263 |
| XXXX | 0.5095436999981757 / 0.5190275000059046 | 0.39466510000056587 / 0.3966039000079036 | 0.3854713000182528 / 0.39043390000006184 | 0.3038101000129245 / 0.30827729997690767 |

Prepared-full median speedups against reference are `1.3564453429022028x`,
`1.2965784875903592x`, and `1.3218719525268103x` for NNNN/NSSN/XXXX. NNNN
tagged median speedup is `1.7900092241256438x`.

The five observations, in measurement order, were:

| Rule / mode | Five observations (seconds) |
|---|---|
| NNNN reference | 0.4933673999912571, 0.5108022000058554, 0.5012425000022631, 0.5116986999928486, 0.5008665999921504 |
| NNNN optimized raw | 0.3849624000140466, 0.37870030000340194, 0.3695447000209242, 0.371841799991671, 0.3755918999959249 |
| NNNN prepared full | 0.3699672000075225, 0.369734600011725, 0.3691046999883838, 0.36827809998067096, 0.36952649999875575 |
| NNNN prepared tagged | 0.2834067000076175, 0.2753550999914296, 0.2801579999795649, 0.2800223000231199, 0.2786326000059489 |
| NSSN reference | 0.4860298999992665, 0.4908020000148099, 0.49794630001997575, 0.4960832000069786, 0.4922091999906115 |
| NSSN optimized raw | 0.39355609999620356, 0.3897220000217203, 0.3928436000132933, 0.3888963999925181, 0.3862032999750227 |
| NSSN prepared full | 0.37962160000461154, 0.3833575999888126, 0.3762292000174057, 0.3775041000044439, 0.380814999982249 |
| NSSN prepared tagged | 0.29238279999117367, 0.2870278999907896, 0.29547519999323413, 0.29787959999521263, 0.2961590000195429 |
| XXXX reference | 0.5190275000059046, 0.5116338999941945, 0.5095436999981757, 0.5086679999949411, 0.5048196999996435 |
| XXXX optimized raw | 0.3917797999747563, 0.3966039000079036, 0.3928075000003446, 0.39466510000056587, 0.39657129999250174 |
| XXXX prepared full | 0.38376029999926686, 0.3890578999998979, 0.3790371000068262, 0.39043390000006184, 0.3854713000182528 |
| XXXX prepared tagged | 0.30827729997690767, 0.30520719999913126, 0.29492149999714456, 0.3038101000129245, 0.3022050999861676 |

## Eight-worker batch

One task per Expert, one complete scenario per task, and `max_workers=8`:

- start method: `spawn`
- task granularity: `(deviating_expert, candidate_rule)`
- reserved/completed/failed calls: `8 / 8 / 0`
- elapsed: `1.8538537999847904` seconds
- throughput: `4.315334898612628` calls/second
- parent peak RSS: `63,647,744` bytes
- worker peak RSS: `32,858,112` to `33,939,456` bytes
- RSS measurement: supported through Windows `GetProcessMemoryInfo`

Using the frozen upper bound of `921,088` calls, the observed batch
throughput gives a mechanical worst-case estimate of
`213,445.31111504883` seconds, or `59.29036419862467` hours. This is a timing
estimate, not a campaign result or an authorization to launch one. The batch
probe used one scenario per task; a future campaign launch decision must
account for its fixed multi-scenario task shape and orchestration overhead.

## Diagnostics

An initial stdin-launched measurement could not create Windows `spawn`
children because Python resolved `<stdin>` as the main script. It produced no
scheduler result and no retry. The valid measurement used a temporary
top-level script with an `if __name__ == "__main__"` guard; that script was
deleted after the probe. No Fit/Validation/candidate/campaign artifact was
created.
