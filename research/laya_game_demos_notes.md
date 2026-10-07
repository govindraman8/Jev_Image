# Research notes: how Laya game demos get fast play (started 2026-09-24)

## madewithlaya.com page 1 (40 of 115 entries)
- Snake laya-mlx @mizorewww x.com/mizorewww/status/2101473552956555427 "50x faster than Jev, <=1GB"
- JevBench 7 open models, Laya last @yankis0x x.com/yankis0x/status/2102045013681008909
- 1000 emails 6 folders, 322M Laya MacBook Air M3: 28.6s, 65.1% acc @NURM_Dima
- tic-tac-toe Unity @unitycoder_com
- ScreenQuest Qwen3.5-4B/MLX + Laya/CoreML + Vision OCR, M3 Max @TremerDaniel
- Snake 30s: Laya score 46 len 52, Jev score 1 len 7 @simplifyinAI x.com/simplifyinAI/status/2102021706613473287
- fighting game World Summit @cristianexer
- Arena survival Jev vs Laya @sreexts x.com/sreexts/status/2102230115245551867
- Pac-Man Laya-MLX @ahmetsoybelli 351 steps x.com/ahmetsoybelli/status/2102014968400888067
- Breakout ~70 dec/s RTX 5080 322M multilingual @eng_ahmd x.com/eng_ahmd/status/2102233195294920886
- Doom deathmatch @hope_rythmn x.com/hope_rythmn/status/2102178634870264308
- Maze 19ms median base M5 vs ~300ms Jev API @developedbyed x.com/developedbyed/status/2101825541728854156
- Tetris bench @tdinh_me: laya-mlx ~84ms but loses 0-3 to Jev, "much more stupid" x.com/tdinh_me/status/2101919182824804560
- Chrome dino @heyxviraj x.com/heyxviraj/status/2102048070649405592
- Tetris ~27ms @brainFnCl (the Laya author)
- @tomhacks laya-mlx tied with custom JAX build; "problem isn't models, it's how you define problem space"
- @tobiaswup Snake: Jev vs Laya-MLX vs Kev-4B
- @0xchewa: Laya beat Jev 43 to 1 wifi off; 1281 moves vs 47; 86.4 vs 3.2 dec/s; 1GB
- @Alex_tra_memory CoreML 99.5% ANE, 3.7ms M5 Pro
- @atomic_chat_hq Tetris built with grok 4.7, 11x faster, 16GB MacBook Air x.com/atomic_chat_hq/status/2102160983409955244
- AXERA-TECH/Laya HF <70ms AX650/AX8850
- litert-community/Laya-Multilingual-LiteRT
- snsk Japanese business bench Jev 97.6 vs Laya 36.9 github.com/snsk/jev-laya-japanese-business-benchmark
- Feishu bench Jev 64/64 vs Laya 20/64; Jev 253ms vs Laya 151ms github.com/Adkid-Zephyr/chinese-workflow-decision-bench
- elcronos: advertised win does not reproduce; tweet_topic Jev .793 vs Laya .632 github.com/elcronos/jev-vs-open-decision-models
- Alexander-Ollman laya-ft Banking77 51.3->79.4 FT vs Jev 80.0
- yibie laya-jev-lab cascade 78% = Jev acc, 1.8x speed
- PerryLink laya-mcp: silent truncation & constant noul fix
- smile-magic/laya-mlx-wzq Gomoku: "model picks white's move from six candidates"; rules in Python
- ipenywis/laya-ultrafast

## brainfunctioncollapse.com/laya (Tetris, author's own page)
- "Code lists the landing spots and does the counting"; each landing spot described as a plain-English sentence, e.g. "The piece leaves one hole under it and makes a small bump on top"; Laya returns P(clean)
- 1,799 decisions and 52 lines in ~1 min; "Thirty a second"; games 95-120 s, 75-100 lines

## brainfunctioncollapse.com/laya more quotes
- "Each place the piece could land is described in plain English and Laya says how the stack would look."
- "Laya decides around 30 times a second, so it flies the bird itself." (flappy bird demo?)
- "21 ms per decision, on a laptop GPU" ; "21 ms measured on an Apple M1 Max"; "27 milliseconds" & "52 cleared lines"
- links: github.com/NandhaKishorM, hf convaiinnovations/laya, arxiv 2503.23303, 2510.01237, x.com/brainFnCl/status/2101782262949835211, skills/laya-integration/SKILL.md

## catalog page 2 (game + runtime repos)
- PromptEngineer48/laya-vs-jev-arena: Snake race + MK-style fight, "same input and typed questions"
- aovestdipaperino/laya-pong: 18.7 ms/frame on Metal, one typed question per frame; other paddle is 3 lines arithmetic
- hama-jp/laya-tetris-finetuning: 421M FT for Tetris; "game enumerates placements with their holes, heights and clears; Laya picks one from text"
- mraad/lunar-laya: CUDA-trained FT lands 30/30 on MLX; 6-bit -58% peak mem
- raboija/PacmanLocalJev: unchanged 322M on Windows CUDA, "planner-intervention counts"
- cv/laya-plays-smb3: DGX Spark, 207 decisions, text obs from RAM, 27.5s game time
- kspviswa/chakravyuha-oss ring maze
- senthilr-nv/laya-warehouse-safety: deterministic shield
- myxamediyar/chunklaya: Laya stops reading after ~200 tokens (attention drop-off)
- afshinm/laya-mps ~32 ms median PyTorch MPS
- yzfly/edgejev ONNX INT8 15.6 ms on 4-core CPU; Jev API 314 ms
- bvolpato/kevala Rust->Wasm WebGPU; r4ai/laya-web ORT-web WebGPU; nvkudva/laya-web 8-bit ONNX 524MB English
- tahby/LayaKit CoreML Swift; MstyAI/laya-mlx-swift; 0x440-1me/laya-unity; 0xBakeer/arbiter; chneau/docker-laya
## catalog page 3
- b0xtch/laya-candle fused Metal kernels M1 Pro; aovestdipaperino/laya-rust; lkarlslund/laya.cpp; li-ming1/laya-zig; receptron/laya (node ORT)
- mizorewww/laya-coreml: 4.98 ms P50 on M3 Max ANE FP16 (short multilingual decision), 2.78x better energy than MLX FP16; Snake 49-50 dec/s
- mizorewww/laya-mlx: <1GB; Snake 60 dec/s on M3 Max
- criticism: abhijay Berkeley prob exam Jev 83.7 / SemIf 61.6 / Laya 31.2; AbhishekDash sysone-bench 751 q Jev sweeps; my_small_room tickets Jev 90-100 vs Laya 30-80; MadhavSz ADK triage Jev 53% 422ms vs Laya 10% 152ms; aaronedell OpenClaw Laya 85% 27.8ms M4 Pro
- shantanugoel: 24h RL on Doom, nowhere near Jev zero-shot
- mraad lander: Laya alone 0/90; with MPC veto 90/90 (veto ~1 in 3 moves)
- NandhaKishorM/laya: "Batched: 7.2 ms/question"
- Kaggle FT notebook: notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb; author: "Base model has limitations"

## hama-jp/laya-tetris-finetuning (Apache-2.0) READ CODE
- 421M English (ModernBERT-large) base convaiinnovations/laya@c5d7873, FULL fine-tune (not LoRA), RLCD-style loop, RTX 3090, ~25 min, 1200 train/200 val decisions
- Teacher = classic heuristic: score = 3*lines - holes - 0.1*agg_height - 0.3*bumpiness - 50*game_over; softmax(score/0.3)
- CODE enumerates every (rotation, column) placement by simulating the real controller; options labelled orientation_XX_column_YY; option text is precomputed RESULT facts:
  f"clears {n}, holes {h}, bump {b}, tot {sum}" (+ " GAME OVER after")
- question: type choice, instructions 'Play this turn of Tetris. Which proposed placement gives the best chance of earning more points before game over? Use the rules and the next-piece queue.'
- state: board_rows (10-char strings, # / .), empty_top_rows, column_heights, covered_empty_cells, current_piece, next_pieces, rules summary, legend
- ONE question per piece, ~34 candidates, 1028 tokens; max_len raised to 1280, head_max_len 896
- KEY: "the old 256 budget cut every option to ~7 tokens per option by build_sequence's head_max_len budget, hiding rotation/column values entirely — root cause of v1-v3 training failure"
- val top-1 agreement with teacher: 84% (epoch1 77.5%); sanity: label-swap follow rate 0.867 over 30 trials
- games (seed 423/424, 180 s, pace 3.0 s min piece age): 22/23 lines, 9100 score, 64/65 decisions, median ~195 ms (p95 293 ms) on RTX 3090; single test 35 ms; 0 stale
- README: not a comparison with Jev; can't attribute gain to training alone

## NandhaKishorM/laya README (official)
- T4 speed: laya(Eng) 39.5 ms 1q, 84.5 ms 5q, 158.6 ms 10q, 771 ms 50q; multilingual 32.8/40.1/72.3 (7.2 ms/q)/337 ms (6.8 ms/q). "Batched throughput reaches 103-332 questions/sec on a single T4"
- predict_batch: "on an RTX 5060 Ti, per-decision latency drops from ~10 ms one-by-one to ~1 ms batched (~9-10x). On CPU, increasing batch size alone may not speed up"
- TileLang fast path laya[fast] (CUDA only, CUDA graphs per (batch,len) bucket); falls back on CPU/MPS
- TOKEN BUDGETS: laya (English) 512 ctx, head_max_len 192 (~320 tokens state); multilingual/typed-decisions 1024 ctx, head_max_len 256 (~768 state). "around 20 options with a short description each, every option is trimmed to fit"; Banking77 77 labels -> 3-4 tokens/label -> 0.425 vs Jev 0.870. Fix: raise agent.cfg["head_max_len"]=512, max_len=1024+, or predict_shortlist, or coarse->fine split
- HONEST LIMITS: "The base checkpoints are near chance on typed-decisions zero-shot -- 0.362 and 0.352 against 0.318 random and 0.461 majority". "Laya is a fast base to specialise, not a zero-shot decision engine." 0.766 is FT'd checkpoint on own train split.
- boolean-word labels bias; negation failures (#377); noul follows labels on English (#156); multilingual position bias on score (#131); act_probability no signal (#185); score questions weakest (SST-5 0.372)
- Long docs: multilingual max_len=8192; 4000-token input ~1.7 s on Apple GPU
- browser-agent FT: element top-1 among ~45 candidates 0.10 zero-shot -> 0.66; task success 0% -> 62% at 17-23 ms/step (cklxx/laya-browser)
- Community: laya-Ascend (Snake & Tetris demos), laya-apple (tc3oliver, MLX GPU + ANE), omp-laya-judge (Snake demo), stuntd
- Jev independently measured 236-276 ms p50 (AbdelStark/jev-benchmarks, nibzard/decision-model-benchmark)

## Official laya-ts/examples/snake.mjs (NandhaKishorM/laya, Apache-2.0)
- state: `Snake on ${W}x${H}, head (x,y) facing DIR, food (x,y), body (x,y) (x,y)...`
- 4 noul questions per tick in ONE systemOne call: `Will the snake die (wall or own body) if it moves ${d} next? ${boardText()}` (state repeated inside instruction!)
- CODE computes exact lethal() and foodPull (Manhattan delta). pick = argmax(pull[d] - 2*danger[d]). VETO: if exact says pick is lethal and a safe dir exists, override with best safe dir. Falls back to pure heuristic after 5 errors.
- => the model is fused with and vetoed by exact code; the game can't lose to a model mistake unless boxed in

## mizorewww/laya-mlx (Apache-2.0, 6.2k stars) Snake — READ CODE (laya_mlx/snake/policy.py, game.py)
- runtime: MLX FP16 native port (encoder+head in MLX), weights aac6fef/laya-multilingual-mlx (default) ; M3 Max 40-core GPU 128 GB
- API numbers (1 short question, end-to-end): English 13.42 ms p50, multilingual 7.39 ms p50; 50-q throughput 146.8 q/s Eng, 395 q/s multi (batch 64); peak mem 944 MiB / 688 MiB
- SNAKE POLICY: code builds a Hamiltonian cycle planner (24x16 board). For each of 4 dirs code computes legal/safe/advance/eats; "preferred" = safe move with max advance.
- compact prompt state = "Safe route: yes. Food reachable through empty cells: yes."  (NO BOARD GIVEN TO MODEL)
- question 'move' choice, instructions "Choose the best safe move toward food.", criteria per dir: "Blocked. Collision." / "Unsafe. Traps the snake." / "Safe. Eat food now. Best." / "Safe. Best route to food." / "Safe. Slower route."
- + 'risk' noul "Is a safe route available?" + 'food' noul "Is food reachable through empty cells?" -> 3 questions batched (batch_size=3), one predict call per move
- shield: if proposed not in safe set, executes highest-prob safe dir. (--unassisted disables shield but still gives planner features)
- seq lengths: multilingual 59-64 tokens, English 66-70 tokens
- infer p50: multi 9.12 ms eager (8.56 compiled+bucket+prefix), English 21.83 ms; complete loop 63.61 moves/s uncapped (46-76 per seed), optimized 75.40 vs eager 70.82 (+6.5%)
- 8,160 moves zero deaths, 4 interventions; raw no-shield control seeds 101-103: survived, scores 9/24/19 vs 20/24/23 shielded
- model comparison 20 seeds x 300 moves: English median score 7, 23.15 ms; multilingual median 10, 9.38 ms
- Doc says: "feature-assisted neural decision demo", "Neither mode establishes that the checkpoint can infer Snake strategy from an unprocessed board." "checkpoint was not trained on Snake"
- compact vs detailed prompt 11.80 -> 9.29 ms median
- opt-ins: compile=True, pad_to_multiple=16, cache_prompts=True (prefix cache of tokenized question prefixes, NOT hidden states); "does not claim to encode the state once and reuse hidden states across questions"
- "The current investigation does not support a further universal 10x speedup"; selected 1.03-1.08x

## laya-mlx BENCHMARKS.md (M3 Max 40-core GPU) latency vs length & questions
- English laya 1q 93 tok: MPS FP32 22.70 / MLX FP32 18.55 / MLX FP16 17.75 ms ; 10q: 95.8 / 103.8 / 80.8 ms; 50q: 489/462/347 ms
- multilingual 1q 91 tok: 13.60 / 10.73 / 10.91 ; 10q 43.9/37.5/32.9 ; 50q 195/159/125 ms (402 q/s)
- FULL CONTEXT: English 512 tok 1q: 64.1 / 64.8 / 49.8 ms; 10q 654/540/420 ms. multilingual 1024 tok 1q: 54.4/51.5/43.5; 10q 528/453/339 ms
- => cost is ~linear in (#questions x sequence length): every question row re-encodes the whole state
- AG News 256 ex: laya 0.957, multi 0.945, typed 0.965 (both backends identical)

## Laya sequence layout (laya/common.py build_sequence)
- "[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]"
- each option capped at 48 tokens; if options overflow head_max_len-16, each cut to max(4,(head_max_len-16)//n) tokens; instructions cut to max(8, leftover)
- state LAST; default truncation keeps FIRST `room` tokens (truncate_left=False) -> tail of a long state is silently dropped
- state_ids can be tokenized once and reused, but encoder still runs per question row

## mizorewww/laya-coreml (Apache-2.0, 1.4k stars)
- ANE bundle aac6fef/laya-multilingual-coreml-ane: CPU+ANE, fixed B1/L96 -> "96-token total limit, including question, options and state. Longer requests raise a capacity error"
- M3 Max: 91-token q padded to 96: compiled MLX FP16 6.94/7.39 ms p50/p95; CoreML ANE FP16 4.98/5.31; ANE W8 4.88/5.23. Energy 0.429 J -> 0.154 J/decision (2.78x); power 61 W vs 31 W
- ANE L1024 graph: actual 1024-token request ~91.7 ms -> "short ANE result does not establish a long-context advantage"
- Snake: 49.1-50.0 decisions/s complete loop (3 sequential calls on ANE adapter), zero deaths, 2 interventions; "does not establish a consistent full-game speedup over compiled MLX"
- 6-bit & 4-bit failed fidelity gate; W8 passes
- general bundles: aac6fef/laya-coreml (512, CPU+GPU), -multilingual-coreml (1024), -typed-decisions-coreml; snake GPU bundle B3/L64

## wdobry/laya-playground (MIT) = brainfunctioncollapse.com/laya source (brainFnCl Tetris/Flappy/Runner) READ CODE
- live site replays RECORDED runs (static/data/run-*.json); local server.py runs real model; "about 30 decisions a second"; Apple silicon, NVIDIA or CPU; 2.3 GB weights
- TETRIS static/demos/tetris.js, checkpoint 'english':
  - code comment: "The model cannot count or compare, so the arithmetic happens here and it gets the conclusion in words."
  - code enumerates every distinct resting place (rotation x column), computes holes, bump grade, lines
  - state (one per spot) = `The piece leaves ${HOLES} under it and makes ${BUMPS} on top.` + ` It completes ${LINES}.`; HOLES=['no holes','one hole','two holes','three holes','many holes'], BUMPS=['no bump','a small bump','a big bump','a tall tower'], LINES one..four lines
  - question: {look: {type:'choice', instructions:'How does the stack look after the piece lands?', criteria:{clean:'flat with no holes', messy:'holes or a tall tower'}}}
  - ONE question per decision, one spot per decision, spots read left to right; piece aims at highest P(clean) so far; drops once all read; gravity doesn't wait (unread spots never considered)
  - MEMO CACHE: "the same sentence always gets the same answer, so it is asked once"  (tiny vocabulary: <=5x4x5 sentences)
- FLAPPY: "Laya is asked WHERE the bird is relative to the gap; the game flaps when P(below) clears a threshold. Asking what to DO comes out inverted, and numbers are not understood."
  - state `The bird is ${band} the gap.` bands far below/a little below/level with/a little above/far above (code quantizes distance)
  - q {where: choice 'Where is the bird relative to the gap?' criteria {below:'lower than the gap', level:'lined up with the gap', above:'higher than the gap'}}; flap if P(below) > threshold
- RUNNER: state "The left lane is blocked by a barrier. The middle lane is empty. ..."; q choice 'Which lane is empty?' criteria [left,middle,right]; stay if P(current) >= stay threshold
- server.py: stock PyTorch `laya` package (Router), MPS on Mac, one HTTP request per decision, serialized by a LOCK ("one GPU, one queue"), keep-alive "20+ requests a second"
- RECORDED RUN METADATA (static/data/run-tetris.json): machine "Apple M1 Max, GPU", laya 0.3.4, checkpoint english, summary {score 52, crashes 0, decisions 1799, per_second 30, median_ms 26.9}; my count: only 32 DISTINCT output vectors across 1,799 decisions; median 26.9 ms, p95 27.8
- run-flappy.json: M1 Max, english, threshold 0.5, score 46, 0 crashes, 1693 decisions, 28.2/s, median 27.2 ms; only 5 DISTINCT output vectors (one per band) -> model acts as a 5-entry lookup table
  - far below->(0.945,.04,.014); a little below->(.816,.146,.037); level->(.056,.918,.026); a little above->(.07,.243,.687); far above->(.017,.034,.95)

## RehmanaliMomin/TetrisGame_Laya (NO LICENSE file; weights rehman-ali/laya-tetris) — BEST CONTROLLED EVIDENCE
- raw-board FEATURES (not placement results): state e.g. "piece: T | turn: flip | shape: width 3, underside 1 0 1 | heights: 4 4 3 3 5 6 6 4 2 2 | steps: 0 -1 0 +2 +1 0 -2 -2 0 | holes: 0 0 1 0 0 0 0 0 0 0 | max height: 6"
- 2 sequential choice questions per piece:
  TURN_Q: "How should the falling Tetris piece be turned before it drops?" {spawn:'keep it as it spawns', right:'turn it clockwise once', flip:'turn it upside down', left:'turn it counter-clockwise once'}
  COLUMN_Q: "In which column should the turned piece's leftmost block land to keep the stack flat and clear lines?" {c1:'column 1 (left wall)', ... c10:'column 10 (right wall)'}
- safety mask optional (swaps a topping-out column for best surviving column)
- RESULTS 5 games x 8000 pieces, mask OFF, M5 Pro GPU (PyTorch MPS, laya pkg):
  teacher El-Tetris script 3198.2 lines, 0.2/0.5 ms; laya-tetris FT 3196.2 lines, never topped out, 79.5% teacher agreement, 36.9/43.4 ms per piece (2 passes);
  laya-base UNTUNED multilingual: 0 lines, topped out after 25 pieces, 5.2% agreement (random 7.0%, tops out after 26); untuned needs mask on 44/100 pieces
- val acc (6000 held-out boards): turn untuned 33.3% vs FT 84.5% (chance 25%); column untuned 10.7% vs FT 82.6% (chance 10%)
- FT: 1 epoch 163k boards, batch 32, ~110 min on M5 Pro; embeddings frozen (197M of 322M params) ; soft targets softmax(score/3)
- design.md: "Base Laya checkpoints are near-chance zero-shot"; "features rather than a raw grid, because a text encoder reads features far better"; lessons: freeze embeddings, split by game, half rollouts from garbage stacks, DAgger optional
- inspired by Okbatti/SnakeGame_Laya (@OwaisBatti)

## PromptEngineer48/laya-vs-jev-arena (MIT, 29 stars) — Jev vs Laya races, READ CODE
- server.py: /api/laya = laya.Router in-process (PyTorch); /api/jev proxies https://api.typesafe.ai/v1/systemone
- ModelAgent (shared/agents.js): PIPELINED loop, workers=3 concurrent in-flight requests staggered 380 ms, minInterval 60 ms; "a ~1.2s round trip becomes a ~3/s decision rate"; decision held until next answer; turn budget caps heading change per answer
- Snake: state JSON = arena size, snake length/speed, nearest_apple {distance_px, direction '<deg> degrees to the left of the current heading', bearing_degrees, is_behind_the_snake, side, shortest_route_goes_through_an_edge}; nothing can die (wrap + pass-through)
  - q steer choice HARD_LEFT/LEFT/STRAIGHT/RIGHT/HARD_RIGHT with long instructions + criteria like 'Ease left. Use when the apple is moderately to the left.'; q sprint noul
  - fairness: same seed; ONE shared pace derived from the SLOWEST side's latency; warm-up before clock
- RESULT (README): "Laya won the snake race 90-50 and took the first fight by K.O.; in the second fight Jev was ahead on health. Laya ~133 ms p50 (13.4 decisions/s), Jev ~962 ms p50 (2.9 decisions/s) — Jev's latency includes my network round trip from India. Asked the snake steering question on 8 fixed apple positions, Jev answered 8/8 correctly and Laya 7/8, with lower confidence. Laya's edge here is speed, not judgment."
- design lessons: "Flapping is a reflex no model can do... the model reads the next gap and code flies."; "Tell Laya one thing per sentence. 'The next gap is near the top. The bird is in the middle.' made Laya answer with the bird's position (0/6); describing only the gap gave 6/6."
- Tetris arena: STATE const 'A Tetris piece is about to be placed. Each question describes one possible placement.'; one request per piece; one noul per DISTINCT placement sentence (dedup -> ~5-12 q not ~30): instructions `Would this placement leave the stack clean? ${t}`, criteria {true:'no new holes and the surface stays flat, or lines are cleared', false:'it creates holes or a tall uneven bump'}; pick max P(true), ties -> lowest landing; timer 3.5 s shrinking to 1.5 s; timeout drops at spawn
- Fight: action choice ADVANCE/RETREAT/PUNCH/KICK/BLOCK/JUMP + commit noul; pace tuned so a kick's startup ~1.15 round trips of the slower model

## aovestdipaperino/laya-pong (Apache-2.0 LICENSE text; GH says NOASSERTION) — CONTROLLED ENCODING EXPERIMENT
- runtime: laya-rust (candle) native server, Metal; English 421M root checkpoint; 18.7 ms p50 per choice on Metal (56% of 30 fps frame); CPU 138 ms (~7 fps); WASM game calls POST /decide once per frame
- state_text: code buckets d=ball_y-paddle_y into 5 words: far above / slightly above / level with / slightly below / far below the paddle + "coming towards the paddle" | "moving away towards the far wall"
- q: Question::choice("Which way should the paddle move to reach the ball?", ["move the paddle up","keep the paddle still","move the paddle down"])
- MEASURED 5 games x 600 frames cap, frames/life: arithmetic 528 (45 returns); model+sentence 528 (45) = identical; random 27; model + "paddle_y 0.50, ball_y 0.79, ball_direction 301 degrees" = 15 frames, 0 returns ("never moves", answers keep still for 0.15 and 0.85 alike)
- options as phrases 5/5 vs ["up","stay","down"] 3/5 root, 4/5 FT
- README: "the geometry is done by the bucketing in state_text, and what is left for the model is the step from 'far below the paddle' to 'move the paddle down'. The demo shows that a typed decision fits inside a 33 ms frame and comes back correct, not that Laya can play pong."

## raboija/PacmanLocalJev (Apache-2.0) Pac-Man on Windows RTX 4070 Ti CUDA FP16
- unchanged laya-multilingual 322M; route search 12 moves deep width 24 describes each direction as blocked / predicted unsafe / safe-slower / safe-best route (planner's preferred move given); 1 choice + 2 noul (safe_route, food_reachable) per move
- execution filter overrides blocked/unsafe; after 10 moves without pellet, recovery picks planner move
- CUDA graphs: short requests padded to 32-token buckets, <=4 graph shapes cached; >256 tokens or batch >4 go eager
- RESULTS seeds 20000-20019: Laya+route guidance 19/20 wins, full decision p50 5.20 ms, 19,453 calls, 3,828 overrides (19.7%); planner only 20/20 wins, 0.96 ms
- Laya inference 4.18 ms median; headless loop 186.8 moves/s
- README: "The planner alone wins more games and runs faster. This demonstrates local typed inference in an instrumented game loop, not that adding Laya improves this planner."

## cv/laya-plays-smb3 (no license) SMB3 World 1-1, DGX Spark GB10
- structured-RAM text observations, fixed rightward movement; Laya controls JUMP via 5-option phase classifier, SAMPLED (seed 96, temp 0.25), stepped mode (emulator pauses for inference)
- 207 decisions, 1,653 frames / 27.50 s game time, 16.79 ms median inference; unmodified weights
- winning seed SELECTED after 2,826 development episodes; "not a reliability or real-time claim"; earlier mushroom run 43.13 ms median (two model calls), 64 continuations none completed

## mraad/lunar-laya (no license) lunar lander, MLX (aac6fef/laya-multilingual-mlx) Apple Silicon
- guidance controller computes desired rotation/power; state text includes "Requested tilt correction: {rotation}. Requested engine power: {engine}." + many numbers (altitude, pad offset, velocities, tilt, desired, fuel)
- 2 choice questions (sequential): "Control lunar lander tilt. Follow the requested tilt correction." {left:'Decrease tilt angle', hold:'Keep current tilt', right:'Increase tilt angle'}; "Control lunar lander engine. Follow the requested engine power." {off,half,full}
- ORIGINAL checkpoint: raw 0/3 landings (all out of bounds) "even though its prompt contained requested guidance labels"; assisted 3/3 but 46.9% of decisions overridden; trajectories identical to baseline; latency 10.1 ms p50 MLX
- baseline controller alone 300/300 at 0.003 ms
- supervised FT checkpoint (CUDA, 3 GPUs): 30/30 held-out raw landings on MLX; Q6 6-bit export -58% peak mem
- related x post: Laya alone 0/90, with MPC veto (~1 in 3 moves) 90/90 (mraad/lunar-mpc-laya)

## smile-magic/laya-mlx-wzq (no license) Gomoku, laya-mlx 0.1.0, aac6fef/laya-multilingual-mlx FP16, tested M4 32 GB
- rule layer scans empty points within 2 of stones, checks immediate five / double threats / short forced-reply traps, keeps <=6 candidates; model gets full 15x15 text board + per-candidate "eligibility. description Shape quality N/100."
- q: choice "Choose White's best eligible Gomoku move. Prefer winning, then defense, then stronger shape." ; then shape-quality gate (>=65% of best in same tactic tier) and picks max model prob among allowed
- input 550-577 tokens; M4 avg whole move ~73 ms, max ~125 ms
- regression 32 positions (4 tactics x 8 symmetries): before input fix raw 18/32, executed 25/32; after raw 30/32, executed 32/32 (improvement from input + candidate handling, no training)
- 4 games vs fixed heuristic black: after fix 1 win, 1 loss, 2 unfinished at cap
- DECISION_FIX.md: explains laya-mlx Snake zero deaths come from safety conditions; upstream test: randomly choosing any allowed action still fills the board (tests/test_snake.py::test_arbitrary_shielded_choices_complete_board_without_starving)

## Okbatti/SnakeGame_Laya (no license; weights OwaisAli10/laya-snake) RAW-BOARD SNAKE, FT
- state: `heading: right | length: 14 | food: 5 up, 11 right | up: free, room 213 open, tail yes | left: body | ...` (code-computed free/room/tail-reachable per dir)
- one choice question per tick over 4 moves; 10 games 15x15, Apple M1 Pro GPU:
  teacher BFS script avg 200.5 (0.2 ms); laya-snake FT 89.9 avg, best 143, 77.7% agreement, 31.8/39.3 ms; laya-base UNTUNED 0.4 avg (3.2% agreement, 31.8 ms); random 1.5 (30.5%)
- FT Colab T4 2 epochs 114k states 41 min; val acc 0.38 -> 0.90; mask never fired for FT model
## kspviswa/chakravyuha-oss (MIT) ring maze
- typed-decisions checkpoint on CPU aarch64 8 cores 7.9 GB, max_len 2048/head 512; each step = 1 question + full state (~783 tokens) -> 1.35-1.51 s per call on CPU; 25/25 easy games reached centre (engine overrules unsure <0.5 moves, forced single-door moves skip model)
- policy mode: 46 questions in one forward pass = 53.7 s; 61 questions OOM (7.3 GB)
- notes temperature_by_options copied (choice:2 -> 1.906, choice:3-5 -> 1.760) flatten distributions

## dante01yoon/laya-jev-arena (no license) — PACED-TO-SLOWER-SIDE Tetris versus + evidence JSONs (Apple Silicon MPS, Jev from South Korea)
- principle: "Code owns rules and arithmetic; the model owns judgement." First version asked model to pick one of six NUMERICALLY described placements -> "arithmetic, not judgement, and one side topped out in 13 pieces." Switched to STRATEGY choice (attack now / build for four / dig out / play safe).
- known limits: "Laya silently truncates past its 512-token context"; "On a cluttered board, both models fail at deduction"
- match-record.json (11 matches, byte-identical payloads, turns paced to slower side): Jev 6, Laya 3, draw 2 (1 void); Laya p50 ~90-95 ms, Jev p50 ~350-412 ms
- game-fitness.json n=30: minesweeper Laya 0.5 (=random 0.5), Jev 1.0; connect four both 1.0 (random .25); tetris Laya 0.333 (random .25), Jev 1.0
- latency-crossover.json (MPS): Laya ~24.5 ms PER QUESTION linear (1q 30 ms, 8q 197, 20q 490, 48q 1198 ms); Jev ~350-420 ms FLAT (1q 364, 20q 347, 48q 421 ms) -> Jev faster above ~16-20 questions/request
- state-length-cost.json (24 noul q constant): Laya 270 ms @255 chars, 557 @515, 933 @853, 1269 @1513, then flat ~1250 (truncation at 512 tokens); Jev 377-530 ms
- context-truncation.json: decisive clue buried deeper; both models wrong (Laya 0.81->0.61, Jev 0.76-0.82, correct <0.5)
- token-cost: tetris turn 699 input tokens, $0.0027 per 92-turn match at $0.042/1M
- korean-checkpoint: multilingual 3/4, English 2/4 (outputs compressed 0.67-0.93)
## htpu/laya-tetris-ai (MIT) Chrome ext on chvin/react-tetris
- heuristic ranks placements, top-k to Laya ("Is this resulting Tetris board state safe and good?"), mix laya_weight*P(safe)+normalized heuristic; if Laya scores within 0.04 -> fall back to heuristic. "Laya ... near random on subjective safe/dangerous judgments ... real level mainly supported by heuristic"; pure heuristic and with-Laya both ~20 lines
## zhangyunting123/von-laya-jev-paint-compare (MIT, from achimala/jev-paint) — painting harness, no results in README; Laya batches capped at 64 questions (server limit); Jev 4 concurrent batches, Laya 1; per-pixel question `What color is pixel (x=.., y=..) in the described image?` 16-colour palette choice
## HarryReidx/jev-laya-tetris (MIT): BFS search in Node engine; Laya on TITAN RTX ~35-45 ms; Jev ~180-250 ms; no controlled results
- LAYA-USAGE.md (Korean): device must be set: 24 questions CPU 1,523 ms vs MPS 493 ms (first wrong conclusion "Laya 5x slower than Jev"); "24.5 ms per question stacks up linearly"; crossover 16 q on this Mac
- "512 tokens -- silently cuts the back"; clue at position 20 and 40 give identical 0.615 -> not reading past
- confidence: tetris 4-landing choice probs 0.286/0.254/0.235/0.226, confidence 0.003; code ignored it and died in 13 pieces; same model answered "at risk of dying now?" noul 0.824 correctly
- table: text meaning (churn) Laya 4/4 Jev 4/4; situational strategy 3/4 vs 3/4; connect-four immediate win (described fact) 100% vs 100%; weighing described results (tetris landing eval) 33% vs 100%; clean deduction minesweeper 50% vs 100%; cluttered deduction both wrong
- minesweeper: Laya outputs trapped 0.55-0.69 band (certain mine called "69% safe"), Jev 0.14-0.96; live prob spread Laya 8.4% vs Jev 21.0%; live ECE Laya 0.476 Jev 0.314
- "The 7 benchmark rows cited ... are all text classification ... measured only the half where Laya wins"
- 4 rules: code makes candidates+consequences; if provable, code should solve; answer must flip with situation; state <1000 tokens natural language (Laya 512, Jev 1024)
- live versus tetris: both 3/4 on situation test; Jev died going for tetris near ceiling; Laya stacked on buried holes

## madewithlaya build pages (X posts, text only; posts themselves not fetchable - HTTP 402)
- atomic_chat_hq (Sep 21): "Local Laya moggs Jev at Grok 4.7-built Tetris ... beat cloud-based Jev at playing Tetris by making decisions 11 times faster, running locally on a 16GB MacBook Air!" no code link
- tdinh_me: laya-mlx ~84 ms on MacBook, "much more stupid. Losing to Jev 3 out of 3 rounds."
- 0xchewa: Snake, Laya (421M) 86.4 dec/s, 1,281 moves vs Jev 1.13.0 3.2 dec/s, 47 moves -> 43 to 1, wifi off, 1 GB
- simplifyinAI: Snake 30 s: Laya 421M score 46 at 86.5 dec/s vs Jev score 1 at 3.2 dec/s (same numbers as 0xchewa, likely same video/tool)
- eng_ahmd Breakout: 322M multilingual ~70 dec/s RTX 5080 Windows laptop; repo github.com/Eng-Ahmd/laya-breakout
- hope_rythmn Doom 1v1: "Jev's calls were sharper, it needed a third of the corrections. Laya just decided twice as often." same state, same typed inputs, same safety params; Laya on MLX
- heyxviraj Chrome dino: Laya MLX on Mac vs Jev API; repo virajbhartiya/laya-vs-jev
- developedbyed maze: 19 ms median local Laya on base M5 vs ~300 ms Jev API
- sreexts arena survival: "game turns each moment into one sentence and asks one typed question"
- tobiaswup Snake: "Jev wins on quality, locals win on speed." "Jev seems to deliver the best quality no doubt!"
- tomhacks: "The problem isn't the models, it's how you define the problem space, design the movements / decisions and optimize for it (batching e.g.)" Laya-mlx tied custom JAX build
- Alex_tra_memory (FluidInference): CoreML 99.5% ops on ANE, 3.7 ms/decision M5 Pro; release FluidInference/FluidUse v0.2.0; HF FluidInference/laya-coreml
## HF FluidInference/laya-coreml model card
- multilingual 322M, unchanged weights; buckets L128/L256/L512/L1024, 32 option slots; FP16 614 MB/bucket or INT8-embedding 448-453 MB
- M5 Pro warm: L128 3.6 ms CPU+ANE / 3.9 ms all units; L512 27.5 ms CPU+ANE / 9.0 ms all units; median 5.2 ms per question (p95 18 ms); 16/16 argmax parity

## Eng-Ahmd/laya-breakout (no license file; README says Apache-2.0) Breakout, RTX 5080 laptop Windows
- upstream laya PyTorch CUDA, torch.compile + fixed-shape CUDA graph (--optimize), multilingual 322M, ~70 dec/s; NO shield, raw top-1 executed
- state JSON with rules + positions + ENGINEERED features in "features" mode: motion, tracking_target, target_x (code-projected wall-reflected intercept), target_vs_paddle: "left"/"right"/"aligned", arrival_ticks
- q choice "Move paddle toward the ball's landing position. Track ball x while ascending. Hold when centered." {LEFT:'Shift paddle left.', STAY:'Keep paddle still.', RIGHT:'Shift paddle right.'}
- no miss-rate numbers published in README
## virajbhartiya/laya-vs-jev (Apache-2.0, 97 stars) Chrome T-Rex, fork of laya-mlx, MLX multilingual on Mac vs Jev API
- deterministic physics planner labels each action safe/unsafe GIVEN EACH MODEL'S OBSERVED LATENCY; prompt e.g. state "Dino runner game. 2 large cacti ahead, 96 px away." q "Choose the best safe action for the dinosaur." jump:'Safe. Clears the 2 large cacti. Best.' duck:'Unsafe. Hits ... Collision.' run:'Unsafe...'
- shields: unsafe->best safe; arrival recheck; emergency shield. "The live shield can keep a dinosaur alive without model answers."
- Jev calling: --jev-inflight default 6 (about four per round trip) => a turn every 5-6 frames; single request => one turn per 22 frames, jump window 18-28 frames -> "Jev died at the first or second one even though it picked the best move every time". Laya keeps 2 in flight.
- HISTORICAL M3 16 GB (heavy swap), 90 s: answer median Laya 33 ms vs Jev 369 ms; model time 19 vs 351 ms; decisions 2,753 vs 197; best score 257 vs 110; deaths 4 vs 11; picked planner's best move 75% vs 100%; shield saves 27 vs 0
- LOCKSTEP (latency removed, 6 frames/decision): deaths 0/0; best move Laya 74% vs Jev 100%; shield saves 30 vs 0
- latest assisted match: equal distance; Jev won both tie-breaks (fewer interventions 18/74 vs 705/69). "show result, not evidence of better model skill"
- course designer: Jev difficulty rose with speed (rank corr 0.47), gaps tightened (-0.56); Laya 0.19/-0.01 ~ random rule (-0.02)
- ~350 input tokens per Jev move request

## RUNTIMES
- bvolpato/kevala (Apache-2.0): Rust->Wasm + WebGPU kernels; Laya int8 pack 479 MB; M4 Max Chrome WebGPU: short request (1 q, 30-45 tokens) 11 ms; "Laya scores 32 states in one pass in 186 ms"; CPU one core ~0.4 s; "Laya is a bidirectional encoder, where a KV cache is impossible; it packs every question of a request into one pass instead" (Kev/SemIf decoder models share one pass over state + resident state caches)
  - decision benchmark (synthetic, 324 cases, batch 1, Q8): laya 75.0% @23.8 ms mean; kev-0.8b 89.8% @30.4; gemma-4-e2b 94.4% @61; semif-4b 100% @363; kev-9b 96.3% @361
- r4ai/laya-web (Apache-2.0): ONNX Runtime Web, WebGPU + Wasm SIMD fallback, multilingual; model.onnx.data 501 MB + embeddings.f16 393 MB (CPU embedding slicing to dodge WebGPU buffer limits)
- nvkudva/laya-web (no license; HF nvkudva/laya-web-q8): English checkpoint 8-bit weight-only (MatMulNBits block int8), 524 MB; WASM only (ORT-web WebGPU MatMulNBits accepts only 2/4 bits; 4-bit argmax collapses to 84.6%); ~340 ms/question short, ~2.4 s at 512 tokens; dynamic int8 quantization FAILS (69% argmax agreement, max dp 0.99) because of activation quantization
- afshinm/laya-mps (MIT, 21 stars): PyTorch MPS, FP32, M5 Pro 24 GiB: median 32 ms (reduced mode 2.11 GiB); minimal 0.74 GiB 171 ms (streams layers from disk); Pong demo; default checkpoint is typed-decisions? ("English model specializes in customer service, invoices, security incidents") 1024 tokens; english 512
- yzfly/edgejev (NOASSERTION, 11 stars): ONNX INT8 dynamic, 4 vCPU Xeon Cascade Lake AVX512-VNNI: laya multilingual int8 324 MB 15.6 ms/question, 3 q 44.8 ms; fp32 1290 MB 32.1 ms; AG News 91.2% (fp32 92.8%), emotion 48.2% (54.0%); int8-static unusable (25.8%); "dynamic quantization results depend on batch" -> use batch=1
  - also lists NanoJev (TianyuCodings/NanoJev, Qwen3-0.6B + scalar head, checkpoint 'unified-games-v1' trained ONLY on Maze/Snake/ViZDoom decisions), kev (jaredpalmer/kev Qwen2.5-0.5B+LoRA), PlayJev (OmniJev/PlayJev); Jev API measured 314 ms median
- tc3oliver/laya-apple (Apache-2.0): MLX GPU + CoreML ANE router; "ordinary Core ML export ran on the Neural Engine without any error and changed up to 85 decisions against upstream PyTorch" -> ANE artifact only after on-machine parity gate; routing: 1 q <=128 tokens -> ANE (typed-decisions L128 9.9 ms ANE vs 12.2 ms MLX forward p50); longer -> MLX (19.2 ms L256, 71.0 ms L1024); several questions -> MLX (batches; ANE runs one at a time); Switchyard demo M4 Max: GPU-only 1,407/1,422 trains late (P99 3.1 s) vs GPU+ANE 0 late (P99 54.6 ms) under 40 req/s bursty load
- aovestdipaperino/laya-rust (Apache-2.0): candle CPU/Metal/CUDA; f16 on accelerators 0.84 GB; "All questions for one state go through a single batched forward pass"; tips: prose not struct; options as phrases (pong 3/5 -> 5/5); "Ask about the world, not about your policy" (noul strongest); act_probability reads 1.0000 always; head_max_len error rather than silently dropping (in rust port)
- b0xtch/laya-candle (Apache-2.0): Rust candle, fused Metal kernels tuned for M1 Pro, optional f16; no numbers in README grep
- lkarlslund/laya.cpp (MIT, 87 stars): ggml C++, CUDA/Vulkan (+CoreML doc), JEV-compatible HTTP server with AUTOMATIC REQUEST BATCHING; CUDA BF16 1.11-2.71x Python throughput on RTX PRO 6000
- litert-community/Laya-Multilingual-LiteRT (Apache-2.0): multilingual, .tflite, wfp16 weights (FC weights fp16 + dequantize, activations fp32) 250.9 MB; S256/S512 windows; Samsung Galaxy S26 GPU FP32 median 50.9 ms/question (256 window); 81/81 argmax parity; INT8 & NPU NOT evaluated
- AXERA-TECH/Laya: AX650/AX8850 NPU3, batch 1, seq 256, up to 4 options; English 69.99 ms, multilingual 27.72 ms, typed 69.99 ms per question; 4-question request 280 ms / 110.9 ms (linear); 481-509 MiB
- vishalmysore/layaForWeb (Apache-2.0, 15 stars): English checkpoint -> ONNX Runtime Web; WASM CPU default; q8e8 ~440 MB (97.9% top-1 agreement over 48 q), q4e8 ~290 MB (WebGPU only works with int4; 10/12 match, max dp 0.274); "three-question call took about 2 to 5 seconds on a 2-core machine". Medium posts (403, not readable) say ~1 second in a browser tab
- jevmodel.org/what-is-laya: repeats README numbers (33-40 ms T4; Jev 236-276 ms); no games, no MLX claims

## CRITICISM
- snsk/jev-laya-japanese-business-benchmark (no license): 40 fictional business questions (12 choice, 12 score, 12 noul, 4 composite) x3 trials: Jev 1.13.0 97.6/100 (39/40 all-3-pass) vs Laya aac6fef/laya-multilingual-mlx FP16 36.9/100 (15/40); Laya on Apple M3 24 GiB MLX, median 41 ms (p95 151), Jev API median 518 ms; 0/48 truncations; "In single Noul, Laya returned true for all 36 trials"
- elcronos/jev-vs-open-decision-models (no license): "Laya's advertised win over Jev does not reproduce": emotion tie 0.587 vs 0.587 (McNemar p=1.0); tweet_topic Jev 0.793 vs Laya 0.632; fin_topic (20 cls) 0.670 vs 0.342; daily_dialog 0.710 vs 0.614; Laya ECE 0.129-0.307 (6-7 options) and 0.610 with 20 options; Laya 31 ms MPS fp32 bs=1 (M1 Max) vs Jev 349 ms p50 via OpenRouter; logistic regression with labels beats all zero-shot
- myxamediyar/chunklaya (Apache-2.0): "past roughly the first 200 tokens, it [decision head] stops paying attention ... A fact buried in the middle of a long document is effectively invisible"; "Find whether something occurs anywhere in a long multi-part input: Plain Laya chance past first ~200 tokens (AUC 0.51)"; fix = chunk into paragraphs so each lands at position 0 of its own forward pass ("the only place the model reads reliably"); default noul misses 2 of 3 true positives; described choice question -> 0.3% FP; PredictionCache keyed by exact input; predict_many with length sorting removed 1.6x penalty
- PerryLink/laya-mcp (Apache-2.0): silent state truncation from the end (st[:room]); options shortened to ~4 tokens; confidence = 1-H/log k (not accuracy); silent CUDA OOM demotion to CPU (10-15x slower); noul answered "false" to 40/40 items (chance) -> asking as 2-option choice with neutral labels: 0.500 -> 1.000 (English), 0.975 (multilingual); truncate-left changed noul 0.0706 -> 0.8341 on decoy doc; char-based token estimates under-reserve JSON 1.45x, CSV 2.15x, CJK 2.1x
- yibie/laya-jev-lab (MIT, M4 Max MLX): 40 Chinese tickets: Jev 78% @588 ms vs Laya 57% @7.6 ms; clear tier 100% vs 75%; cascade at confidence 0.60 threshold: 55% escalation, 78% accuracy (=Jev), 327 ms mean (1.8x); noul "reliable for facts, useless for judgements" ("Does the customer request a refund?" 0.996 but politeness 0.003-0.277); confident-and-wrong failure; confidence-accuracy not monotonic
- dhruvmehra/jevbench (MIT) n=500: agnews Laya 90.6 vs Jev 84.3 (85.8 new desc); sst2 Laya 92.0 vs Jev 95.4; banking77 Laya 38.2 vs Jev 76.4; Laya p50 59/41/130 ms at parallelism 1 on arm64 Mac (16-24 ex/s); Jev p50 ~380 ms, 18-19 ex/s at PARALLELISM 8 via OpenRouter
- Adkid-Zephyr/chinese-workflow-decision-bench (MIT): 64 Feishu scenarios single choice Jev 64/64 vs Laya 20/64 (31.25%); Laya multilingual on M4 GPU 151 ms median vs Jev 253 ms
- lzero07/laya-zh-eval (MIT): 20 Chinese skill-routing requests (chance 0.077): multilingual 0.70 @192 ms, typed 0.65 @844 ms, english 0.60 @667 ms; confident-wrong (multilingual 4 cases up to 0.945)
- Alexander-Ollman/laya-ft (no license): Banking77 1,001 FT examples: Laya 51.3% -> 79.4% vs Jev 80.0%; 5 workflows FT Laya beats Jev on 4 of 5 (injection 94.8 vs 78.7; spam 98.1 vs 90.8; emotion 65.9 vs 49.7; counterfactual 90.0 vs 86.5; routing 75.0 vs 79.9); moderation FT raised false alarms on XSTest 24.8% -> 65.2%/47.2%

## DOOM repos
- shantanugoel/laya-doom (no license): frozen Laya encoder + small trained head; ViZDoom reports who is on screen -> one keyword line incl. scripted route ("Route: LEFT"); "the head copies the teacher". (x post: 24 h RL training "nowhere close to jev 0-shot")
- dylanbstorey/laya-doom (no license): ViZDoom defend_the_center; laya-typed-decisions fp32 MPS via afshinm/laya-mps server, M1 Max 32 GB
  - "Python computes the observation. The model chooses the action." state.py computes bearing, distance, crosshair-on-enemy -> English
  - PHRASING TEST: "bearing +6 degrees right" -> wrong (conf 0.0115); "slightly LEFT of your crosshair" -> correct; best turn phrasing 100% on English states vs 57% on degrees-and-coordinates states
  - 2 questions: fire noul (balanced acc 97%), turn choice left/right/hold/advance/scan (100% on 4-option fixture); threat score question CUT (moved +0.17 on 0-2 scale as health 100->5)
  - latency p50: 2-q battery 61-80 ms; single 32.1 ms; cold 686 ms; --question-batch-size 4: ~96 -> ~62 ms; cadence 4 tics = 114.3 ms = 8.75 dec/s; <=1 request in flight, skipped not queued
  - latency cost: paused world +13 score 14 kills vs real time +1 2 kills
  - hold meaning "no enemy in sight" made model sit still 88% -> redesigned options
- JakkNaj/laya-dino (MIT code; checkpoint Apache): English checkpoint, encoder frozen, decision layers trained on 5,760 physics-labelled states (1,440 held out): 95.42% held-out; mean score 808.0 -> 915.8 (5 paired games, 800-decision cap); ~40 ms warm MPS; no heuristic overrides; lockstep (6 physics frames per decision, "Inference time never becomes a physics jump")
- cohenom/laya-snake (no license): raw board JSON + danger_<dir> booleans; English on CPU ~90-100 ms/tick (45 ms isolated); raw pick immediately fatal ~10% of 60 ticks even with danger flags; "visibly wanders"; safety override

## wdobry/laya-playground skills/laya-integration/SKILL.md ("the traps found building this site")
- "Ask what the text says, not what to do about it. 'Where is the bird relative to the gap?' produced clean graded probabilities; 'Which way must the bird move?' came out inverted on every checkpoint"
- "Put the state into words, never numbers. Given 'Bird altitude: 20. Gap altitude: 60.' no checkpoint could tell which was lower. If a decision depends on a comparison, threshold or sum, compute it in code and hand Laya the conclusion ('the bird is far below the gap')."
- describe each option; keep option lists short (48 tokens per option cap, 192/256 shared); keep state short & front-loaded (512 English / 1024 others, cut from end); try 2-3 phrasings ("blocked by a barrier" >> "blocked by a train")
- measured one question ~34 ms English / 21 ms multilingual on M1 Max GPU; CPU 139 / 58 ms; "Ten questions in one call cost about 7-16 ms each"; warm up at each new batch shape
- multilingual ships UNCALIBRATED (temperature 1.0)
- 500-example independent run: news topic 93%, SMS spam 96% (level with hosted); emotion 45% (hosted 53), prompt injection 65% (71), 5-level star rating 35% (70); ECE 0.05-0.06 easy, 0.34-0.40 hard; 35-66 ms/question

## trungdq88/jev-tetris (Tony Dinh = @tdinh_me; no license; 15 stars) — the "Laya loses 3 of 3" Tetris bench
- code enumerates every legal placement, describes outcome in words; Jev: ONE request per piece: state JSON (board rows, heights, stack_height words, holes words, surface words, pieces) + choice over ALL placements with long priority instructions + 3 extra questions (strategy choice, board_health score, next_piece_fits noul)
- Laya (512 ctx) gets one-paragraph board description + only the 6 best placements PRE-RANKED by code heuristic, each "column 10 vertical: clears four lines, no holes, stack gets lower"; instructions "Pick the best placement: clear lines, avoid holes, keep the stack low and flat."
- RESULT (PyTorch 4-core CPU): versus: Jev wins at 0:14; Jev 7 lines/27 pieces, 232 ms/move; Laya 0 lines/19 pieces, 614 ms/move, 2 missed. Lockstep Laya alone: topped out after 53 pieces, 5 lines, picked heuristic's top option only 16/53
- "probabilities over the six candidates are nearly flat (entropy confidence 0.01 to 0.1), it prefers whichever options are listed first, and it does not favor the 'clears four lines' option even when it is the only one that clears anything." "On Apple Silicon with laya-mlx ... tens of milliseconds, which removes the latency handicap but not the decision quality."
- Jev ~220 ms/move in these games
## fchange/tetris_for_laya (MIT): raw board, keyboard-level action choice (LEFT/RIGHT/ROTATE/DOWN/WAIT), no planner: multilingual MLX topped out after 16 pieces, 657 inputs, 0 lines; proposed RIGHT 561 times; legality guard replaced 266 blocked inputs
## YouTube "Laya-MLX vs Jev: fast answers, losing moves?" https://www.youtube.com/watch?v=U1buxRDB8jI (not watched)
