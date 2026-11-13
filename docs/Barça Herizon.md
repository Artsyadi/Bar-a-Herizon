# Barça Herizon · Technical document

More than a Hack 2027, Phase 1 · possession labelling of BAR-ATH from tracking data
Code: https://github.com/Artsyadi/Bar-a-Herizon

## 1. Summary

We label every frame of the tactical video as home possession (`H`), away possession (`A`) or dead
ball (`D`) from the provided tracking data alone. The system has four parts:

1. a code-only cleaner for the automatic tracking (players and ball);
2. a **dead-ball model** that sees only the players;
3. a **possession model** that sees players and ball;
4. a **two-layer Viterbi segmenter** that turns frame probabilities into possessions.

Both models are gradient-boosted trees (LightGBM) trained on seven external matches (IDSSE) whose
ball tracking we degraded to look like the provided data. Nothing was trained, tuned or calibrated
on the provided game, its video or its ground truth.

On an external match that we held out from every decision and scored once at the end, the system
reaches **spellF1 0.577** (macroF1 0.887), against 0.219 for the organisers' baseline on the same
match.

## 2. Approach, and why

**Tracking data, not video.** We work from the tactical ATD. It is one record per submission frame
(video frame = ATD `frame_id` − 1), already in pitch coordinates, and gives every player's position
directly. The broadcast recording runs on its own timeline (cuts, replays, 164,488 frames) with no
mapping to the tactical frames the submission is indexed to. Learning possession from video would
also need labelled video, and training on the provided video is not allowed. The tracking is
noisy, but noise in coordinates can be handled with code, which is where we put the effort.

**Two questions, two models.** "Is the ball in play?" and "which team has it?" depend on different
evidence, so we answer them separately:

- **Dead ball is decided from the players only.** When play stops, the players show it: both teams
  slow down together, shapes change (walls, kick-off formations, players gathering in the box or at
  a touchline). Using players only means the dead-ball decision does not depend on the ball, which
  is missing on about 41% of the graded frames and which, in our external data, freezes in place
  during dead play. A ball-based dead-ball rule would learn that freeze rather than football.
- **Possession is decided on live frames only**, from the ball and the players around it, plus
  team context that still works when the ball is missing (team compactness, movement towards goal,
  how high the team stands).

**Possessions, not frames.** The ranking metric matches whole possessions, so frame-by-frame
classification is not enough: it flickers and creates more than twice as many possessions as
exist. A segmenter that charges a cost for every switch produces clean possessions and was the
single largest improvement (spellF1 0.332 → 0.553 on our development matches).

## 3. Assets: used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (`_atd.xml` + `_atd.txt`) | **Used**: the only input of the system |
| Tactical video | Watched to understand the match and possession; not an input and never used for training or for checking output |
| Broadcast video and its ATD | **Dropped**: separate timeline, no mapping to the submission frames |
| Homography | **Dropped**: the ATD is already in pitch coordinates |
| Smart Tagging | **Dropped**: we concentrated on the tracking; not used in this submission |
| Public 5-minute ground truth | Pipeline check only: frame alignment, team mapping (first metadata team = `H`), label meaning. Never used for training, tuning or any choice |
| Organisers' scorer, validator and baseline | Used to score external matches, validate the file and compare |

## 4. Training data

All labelled data comes from other games. No label of any kind on the provided game was used.

| Dataset | Matches | Labels and who produced them | Amount | Use |
|---|---|---|---|---|
| **IDSSE** (Bassek et al., 2025; CC BY 4.0) | 7: Bundesliga 2022/23 (J03WMX, J03WN1) and 2. Bundesliga 2022/23 (J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) | Per-frame ball status (alive/dead) and possessing team from the official tracking data of the data provider (Sportec Solutions for the DFL, TRACAB optical tracking). We converted them mechanically: `D` where the ball is dead, otherwise the possessing team. No person on our team labelled anything | 1,002,644 graded frames (about 11 hours), 3,061 possessions | 6 matches for training and development, 1 (J03WPY) held out as the final test |
| **Metrica Sports sample data** | 3 anonymised matches | Derived from Metrica's published event data | 429,923 graded frames, 1,038 possessions | Comparison only; never trained on |

**Making the external data look like the provided data.** The IDSSE ball is almost always visible;
in the provided tracking it is missing on about 41% of graded frames (DATA.md). We therefore made
five degraded copies of every IDSSE match: ball frames removed in gaps, with ball availability
drawn around 59% (between 50% and 70%) and gap lengths taken from the real observation gaps of the
Metrica raw tracking. The players were left unchanged. Every model and choice uses all five copies
and reports the mean and the worst copy.

## 5. The pipeline

**Cleaning** (all thresholds from physics or external data, none from the provided game):

- duplicate tracks merged, impossible jumps and speeds removed (players above 12.5 m/s, ball above
  its physical limit);
- player speed recomputed from positions (centred difference over ±2 frames), because the provided
  speed channel's declared unit does not match its values and the external data has no speed;
- the ball gets a status per frame: observed, estimated across a short gap, or unavailable; nothing
  is invented across long gaps.

**Features** (379 columns, listed one by one in `docs/features_list.csv`):

- dead-ball model, 66 player-only values (264 columns with time windows): share of slow and running
  players per team, both teams slowing together, deceleration, team width, depth, compactness and
  position, defensive line height, kick-off shape, tight groups (walls), players near a touchline
  or in a penalty area, and how players move relative to their nearest opponent;
- possession model, 37 values (115 columns): ball state and motion, nearest player per team and
  the distance margin between the teams, players within 2 and 5 m of the ball, ball–carrier
  agreement, which team has been nearer over the last 1 and 3 s, the last reliable margin before a
  ball gap, and team context;
- every value also as 1 s and 5 s averages and a 2 s change; windows never cross half-time;
- missing inputs stay missing (never zero); a feature is left blank when fewer than 7 players of a
  team are tracked.

**Models.** LightGBM, 400 rounds, learning rate 0.05, 31 leaves, minimum leaf 200 (dead ball) / 500
(possession), fixed before any result and not tuned. Trained on every 5th frame. The dead-ball
model uses one copy per match (its inputs do not depend on the ball); the possession model uses all
five.

**Segmenter.** Two layers, each a two-state Viterbi with a cost per switch, in log-odds units:

1. dead vs live from the dead-ball probabilities (cost 64);
2. inside each live interval, home vs away from the possession probabilities (cost 8), decided
   afresh in every interval.

Both costs were chosen on the development matches from a 5 × 5 grid fixed before running.

## 6. The possession rules we implemented

We wrote our own possession rulebook before building, and the system follows it:

- **Dead ball** is a stoppage, never a possession, and comes from the players only. Frames outside
  the two halves are `D`. The file never contains `X`.
- **Restarts.** After every dead spell, possession is decided afresh from what happens after the
  restart. Nothing carries across a stoppage, so `H` → `D` → `H` is always two possessions. We do
  not model who "owns" a dead ball (the ground truth does not ask for it).
- **Possession changes on control, not on contact.** A touch, a deflection or a won header does not
  change possession by itself. The labels we learn from mark the team the data provider considers
  in possession, and the segmenter enforces the rule: a switch has to be paid for by sustained
  evidence, so a brief contact by the other team is absorbed.
- **Loose and contested balls stay with the last team in control** until the other team's evidence
  is strong enough to pay for the switch.
- **No minimum duration.** A short possession survives when its evidence is strong enough; there is
  no fixed length anywhere.
- **Ball not visible.** Ball features are left blank, never guessed; the model then relies on team
  context and on the last reliable ball–player relationship before the gap.

## 7. How we validated

- **Development:** 6 IDSSE matches, 6-fold leave-one-match-out: every match is predicted by models
  that never saw it. All choices (segmenter costs, keep or drop a change) were made on these folds.
- **Final test:** J03WPY, chosen before any result as the most typical IDSSE match by label
  statistics, scored **once** with the frozen system. All five degraded copies, mean and worst.
- **Comparison:** the 3 Metrica matches, another provider never used for training or choices.
- **Baseline:** the organisers' reference (nearest player + ±50-frame majority vote), reproduced
  exactly (identical output to their code) and run on the same matches.
- **Freeze:** models, configuration and code fingerprinted (md5) before the test; retraining gives
  byte-identical models.
- **Leakage test:** feeding the dead-ball model a shuffled ball, or another copy's ball, leaves its
  predictions identical.

## 8. Results

spellF1 decides the ranking; macroF1 is the tie-break. On one full match, spellF1 differences below
about 0.04 are noise.

| Data | Our spellF1 | Our macroF1 | Baseline spellF1 | Baseline macroF1 |
|---|---|---|---|---|
| Development, 6 matches × 5 copies (out of fold) | 0.553 | 0.875 | 0.216 | 0.457 |
| **Final test J03WPY**, 5 copies (scored once) | **0.577** (worst copy 0.568) | **0.887** | 0.219 | 0.458 |
| Metrica, 3 matches (never used) | 0.684 | 0.877 | 0.427 | 0.806 |

On the test match the dead/live decision reaches F1 0.960 (live as positive class) and, on live
frames, the right team is predicted 87% of the time. The test score agrees with the development
estimate, so our development process did not overfit noticeably.

**What the models rely on.** The dead-ball model leans mostly on the share of slow players and on
both teams slowing together (about 70% of its decisions), then team shape. The possession model
leans on which team has been nearer the ball over the last 3 seconds and on team context: the team
in possession spreads out and moves forward.

## 9. What we tried and did not keep

We kept a change only if it raised mean development spellF1 by more than 0.016 (the 0.04 noise
band over six matches) and improved at least 5 of the 6 matches; the rule was fixed before testing.

| Change | Result | Decision |
|---|---|---|
| A change-of-possession detector making switches cheaper where a change looks likely | +0.009, 5 of 6 matches | dropped |
| Extra training weight on frames within 1 s of a possession change | +0.012, 5 of 6 matches | dropped |
| Labels derived from event data for IDSSE | replaced by the provider's per-frame ball status and possession flags, which are frame-exact | dropped |
| A ball-based dead-ball rule | the ball freezes during dead play in the external data, so such a rule learns a data artefact | dropped in favour of players only |

## 10. Limitations: where it fails

- **Short possessions.** The system predicts about 25% fewer possessions than there are. Most
  possessions shorter than 2 seconds (30% of all) are missed; inside one that lasts under a
  second, the possession model favours the right team on only 41% of frames, because its
  strongest evidence looks back over the last few seconds. On development data, catching these
  perfectly would be worth about +0.18 spellF1.
- **Dead-ball boundaries.** Short stoppages are sometimes missed and some dead spells end late;
  perfect dead/live boundaries would be worth about +0.08 on development data.
- **Noisier input than in training.** Our degradation removed ball frames only. The provided
  tracking also has player gaps, fragmented tracks and identity switches, which the models did not
  see in training. Half of the possession model's evidence is team context that needs at least 7
  tracked players per team.
- **Timing convention.** In the external data the possession flag switches about 1 second after
  the nearer team changes. If the provided ground truth times its switches differently, our
  changes will be systematically early or late, and spellF1 is sensitive to timing.

With more time we would add possession evidence that reacts within a second (who is within a metre
of the ball right now, a change of ball direction at a player), degrade the players as well as the
ball in training, and refine dead-ball boundaries with the ball inside the window the players
define.

## 11. Reproduction

```bash
pip install -r requirements.txt                       # Python 3.10, pinned versions
python scripts/make_submission.py                     # data bundle -> outputs/Barça Herizon.csv
```

The data bundle goes in `more-than-a-hack-2027/data/` (or `MTAH_DATA_DIR`); see the README. The
script checks the frozen models and configuration against their fingerprints, reads and cleans the
tactical ATD, builds the features, applies the two models and the segmenter, writes the file and
runs the organisers' validator. Retraining from the external data is described in the README.

## 12. Compliance with the rules

- Labels are produced by an algorithm; no hand labels or hand corrections on the provided game.
- No use of the private ground truth; the public 5 minutes served only to check the pipeline.
- No training, fitting or calibration on the provided game or its video; all training data is from
  other games.
- No human-produced data about the provided game from any source.
- The repository contains no match data and regenerates the submitted file exactly.

## 13. Data and acknowledgements

- Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal and event
  data in elite soccer. *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
  (CC BY 4.0). Data owner: Deutsche Fußball Liga (DFL).
- Metrica Sports sample data, https://github.com/metrica-sports/sample-data
- BAR-ATH match data and evaluation tools: More than a Hack 2027 organisers (Barça Innovation Hub,
  Metrica Sports, Mobile World Capital Barcelona). The match data is licensed for the competition
  only and is not included in our repository.
