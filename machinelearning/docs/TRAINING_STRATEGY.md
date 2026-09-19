# Training strategy: teach tactics, then improve through search

Investigation date: 2026-09-18. Recommendation only; the training implementation and champion were not changed.

The next experiment should teach the network to recognize and answer threats using engine-generated examples. Then use search to generate stronger training targets and continue learning across evaluations. More short continuations from the same champion are unlikely to address the tactical weaknesses found below.

**What the investigation established**

The inspected champion is `6af31c18-55cb-4213-be55-4fe9fcb4924d`, generation 4, episode 12,625. The database covers recent continuation attempts; its first champion was imported at episode 12,250, so this analysis does not reconstruct the original training run.

The latest completed standalone random evaluation, `abb2347e-28c2-4b80-8c2e-8f4745015436`, used seed 20260921 and 1,000 games on 7×7, without a turn limit. It recorded 968 wins, 22 draws, and 10 losses: 97.9% match score, where a draw counts as half a win. The user's concern is these occasional losses and avoidable tactical mistakes, not an inability to beat random on average.

A read-only metadata snapshot at 00:22:03 UTC contained 92 attempts, 24,120 training episodes across those branches, and 812,000 completed evaluation games. Training was active, so these totals will grow. Evaluation games and training episodes do not have equal computational cost; these are activity counts, not a wall-clock ratio.

Every recorded exploration metric was epsilon 0.05. A fresh random seed does not restart the exploration schedule: `train._epsilon` uses the restored lifetime environment-step count. Each branch also creates an empty replay buffer. Among saved candidates 125 episodes beyond their starting checkpoint, warm-up 2,000 yielded only 849–890 optimizer updates; warm-up 512 yielded 2,337–2,380. New seeds change sampling, but the branches remain mostly greedy continuations of the same policy.

Recent longer branches regress on their own shared random screening suites:

| Attempt prefix | Added episodes | Learning rate | Champion match score | Candidate match score |
| --- | ---: | ---: | ---: | ---: |
| `f5424b6d` | 500 | 0.00005 | 97.97% | 95.80% |
| `3dabec5d` | 500 | 0.005 | 98.37% | 92.55% |
| `9a7d6c81` | 1,000 | 0.00005 | 98.07% | 94.35% |
| `eadce78a` | 1,000 | 0.00005 | 98.13% | 93.88% |
| `83a97059` | 1,000 | 0.00005 | 98.10% | 94.37% |

Each table entry uses 3,000 evaluation games per checkpoint. These attempts trained against 80% frozen champion and 20% random. In `83a97059`, the learner's mean score difference against the frozen opponent declined from −0.319 in the first 250 training episodes to −1.320 in the last 250, while mean logged loss decreased from about 0.060 to 0.040. Those training statistics include exploration and different sampled games, but they do not support the idea that the learner was improving against the champion while merely sacrificing random performance.

**A tactical diagnostic of the actual weights**

I generated two families of sparse 7×7 positions and evaluated the champion greedily on CPU. Each family contains 25 possible interior centers × 4 missing sides × 10 random border-filler arrangements × 2 player seats = 2,000 cases. Seed: 20260918. Counts allow the indicated player to act after six or seven moves. Cases are constructed motifs, not a representative sample of full games or necessarily unique boards.

| Motif | Champion selects the tactical gap |
| --- | ---: |
| Three friendly dots surround an enemy dot; the fourth captures it | 910 / 2,000 = 45.50% |
| Three enemy dots surround a friendly dot; occupy the gap before the enemy does | 943 / 2,000 = 47.15% |

A separate Go diagnostic replayed legal alternating prefixes through the existing engine and verified all 4,000 boards. All 2,000 capture targets scored exactly one point. All 2,000 blocking targets occupied the gap. In all 1,057 missed-block cases, the champion's chosen move scored nothing and the opponent could capture on the next move by filling that gap.

For example, with the learner as A and opponent as B:

```text
       0 1 2 3 4 5 6
    0  . ! . . . . .
    1  B A B . . . A
    2  . B . . . . .
    3  . . . . . . A
    4  . . . . . . .
    5  . . . . . . .
    6  . . . . . . ?
```

A can protect its dot at (1,1) by playing `!` at (0,1), connecting it to the border. The model instead chooses `?` at (6,6). B can immediately capture at `!`. The model assigned approximately −0.88 to the blocking move and +1.13 to its chosen move.

These checks prove immediate tactical consequences, not optimal full-game play. A sacrifice or delayed capture can sometimes be correct. They also do not establish that these motifs caused the recorded random losses; reconstructing those losses is a separate diagnostic. Nevertheless, the results expose a concrete weakness hidden by aggregate win rates.

The temporary reproduction files from this session are `/private/tmp/dots_cordon_tactical_probe.py`, `/private/tmp/dots_cordon_tactical_probe.json`, `/private/tmp/dots_cordon_tactical_cases.json`, and `/private/tmp/dots_cordon_validate_probe.go`. The Python probe runs from `machinelearning/` with `PYTHONPATH=.` and the project virtual environment; the Go validator runs from the repository root and has a ten-second execution watchdog.

**Why the current training can leave this weakness**

Defense is already rewarded: `self_play.py` subtracts the opponent's actual capture from the learner's transition reward. The same-player, two-move transition and Double DQN target are intentional; there is no apparent missing opponent penalty in the inspected implementation.

However, that penalty only arrives when the opponent actually exploits the mistake. Random opponents miss most specific threats, and a single frozen opponent supplies a narrow collection of responses. The learner never receives explicit information about the alternative moves the opponent could have played. Uniform replay and a one-transition target provide no special treatment for rare but decisive mistakes. This is a plausible explanation supported by the diagnostics, not a proven attribution to one algorithmic component.

The training objective is also different from the gate: discounted capture differential plus a terminal bonus is not identical to maximizing match score or minimizing loss probability. A larger terminal bonus does not make these objectives identical. Adding an arbitrary penalty for every threatened dot could suppress valid sacrifices without teaching accurate game evaluation.

**Recommended first experiment: learn reliable tactics**

1. Build a durable tactical validation set before training on these patterns. Include captures, saving one or several dots, escape connections, double threats, dead territory, and positions where sacrificing a dot is correct. Replay positions through the real engine. Use exhaustive endgame search to obtain genuinely optimal move labels where few legal cells remain; label shallow-search conclusions as shallow, not globally optimal.
2. Generate a separate training corpus of approximately 20,000–50,000 varied positions as an initial budget. For each legal move, calculate immediate captures and the largest capture available to an opponent's reply. This teaches consequences even when the actual opponent would miss them. Include quiet positions so the network does not learn that every board contains a threat.
3. Train auxiliary per-action capture/threat predictions on the existing convolutional representation. Train a move-selection head on solved positions or verified teacher choices. Do not substitute a one-ply tactical score for the full-game Q target and call it optimal strategy. For tied optimal moves, supervise the acceptable set rather than an arbitrary first action.
4. Apply all eight rotations/reflections of square-board positions consistently to boards, masks, actions, and labels. Split training and validation by original position/trajectory and symmetry family before augmentation; transformed copies must stay in the same split.
5. Compare champion initialization with a fresh initialization. Keep the 64-channel, three-block model initially to isolate the teaching method. Use a fresh optimizer when changing to a supervised objective rather than carrying incompatible optimization history by default.

The first milestone is near-perfect performance on held-out elementary forced-defense and capture tasks, followed by a measurable reduction in full-game tactical errors. A proposed gate is at least 99% accuracy on independently verified, unambiguous basic tactics, with separate reporting for each family and seat. This is an experiment acceptance target, not a predicted outcome or proof of perfect play.

A full-game defender should not blindly block every threat or always take the largest immediate capture. It should preserve winning outcomes and make justified trades. Solved endgames and search labels make that distinction teachable.

**Next: search supplies stronger targets; the network learns them**

Use a small search player as a teacher, starting with exhaustive immediate replies and solved late-game positions. Benchmark it before assuming it is stronger than the champion. If it is only better on local tactics, restrict its authoritative labels to those tactics and use deeper search or rollouts for general positions.

Then iterate: play games, search the learner's encountered positions, add improved move targets to a retained dataset, train the network, and repeat. Include the learner's own mistakes in the next dataset. This is the basic idea of [Expert Iteration](https://arxiv.org/html/1705.08439v4): search discovers stronger decisions and a network learns to generalize them. Its published results are evidence for the approach, not a performance guarantee for Dots Cordon.

If broader planning remains limiting, extend to a policy/value network with Monte Carlo tree search. Use terminal win/draw/loss targets (+1/0/−1) for the main value objective and keep score or tactical prediction as auxiliary targets. [AlphaZero](https://arxiv.org/html/1712.01815v1) provides the policy/value and search-training pattern; a modest local experiment should be sized from measured simulation throughput, not its original compute budget.

Search can be a training teacher while the deployed model still picks moves without search. Evaluate that network-only player separately from a search-assisted player so improvement is attributable to the model.

The current gRPC API has no clone, undo, or arbitrary-state reset operation. Efficient search therefore needs a runner-side state-copy/search adapter around the existing Go engine, including scores and turn-limit state. Benchmark this before committing to large search budgets. A second, unverified implementation of the rules would weaken both labels and evaluation. No engine production changes were made during this investigation.

**Change the learning loop and the acceptance criteria**

Keep the active learner, optimizer, and recent experience across routine evaluations. Keep the accepted champion separately for comparisons and serving. A failed promotion should not automatically erase the learner's progress. Sustained regression can stop an experiment, but checkpoint acceptance and experience collection should not be the same control loop. AlphaZero likewise used a continually updated learner; that design choice alone does not guarantee stable DQN training.

Use varied opponents after tactical pretraining: for example, an initial experimental mixture of 25% random, 25% tactical/search teacher, and 50% sampled recent and historical policies. Refresh the pool periodically and retain selected older styles. These percentages are starting hypotheses. Track opponent-specific performance and adjust based on coverage, not just aggregate score. Explicitly use a phase-relative exploration schedule if remaining with epsilon-greedy DQN; reseeding alone is insufficient.

For a lower-effort DQN control, first test continuous replay plus symmetry augmentation, then separately add three-transition returns and prioritized replay with importance weighting. A three-transition return here spans up to six individual placements. These changes target experience reuse and delayed credit; [Rainbow's ablations](https://arxiv.org/pdf/1710.02298) support investigating them, but were conducted on Atari, not this game. They are a comparison arm, not a substitute for tactical validation.

Use cheap development evaluations to detect large changes, and reserve large fresh suites for a selected candidate. A practical initial development budget is 200–500 games per opponent every 500–1,000 training episodes, plus the fixed tactical set. Final validation should include both seats against random, a tactical defender, several historical checkpoints, and the current champion. Compare multiple training seeds and record compute as well as episode count when comparing search-based and DQN methods.

Report random loss rate, draw rate, match score, per-seat results, tactical mistakes per opportunity, and regret on solved endgames. Saving every loss's move sequence enables exact replay and counterfactual diagnosis; the current evaluation tables store aggregate counts and seeds, not per-move traces. Keep development suites separate from final test suites.

An initial reliability milestone could be fewer than 0.1% losses against random while maintaining match score, confirmed on a fresh large suite. Zero observed losses is evidence, not proof of a zero true loss rate. Do not claim independent-game confidence intervals from the existing paired-seed protocol without accounting for the pairing; retain per-pair results or use independently seeded games for that calculation.

The first implementation milestone should therefore be the tactical dataset, labels, and benchmark, followed by a small supervised training comparison. Its purpose is to establish that the network can learn protection and capture reliably before spending another large budget on champion challenges.
