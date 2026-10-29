# A Perception-Aware Framework for Evaluating Counterfactual Passing Options in Professional Football

Honours research project, School of Computer Science and Applied Mathematics, University of the Witwatersrand.

**Author:** Raeesa Lorgat
**Supervisors:** Dr Pravesh Ranchod and Dr Branden Ingram

Counterfactual passing models judge a pass by comparing it with every other pass the player could have made. Almost all of them assume the passer can see every teammate at once. This project replaces that bird's-eye view with a movement-based vision cone and a model of short-term memory, and asks how much the verdict on a pass depends on which options the player is assumed to have had.

**Live demo:** [What Else Was On, an interactive pass explorer](https://claude.ai/artifact/Kam5beB97gupLUeEmWQkUX). Pick any of the 98,652 analysed passes, see which teammates were in view, and play out what might have happened if the ball had gone somewhere else.

---

## The framework

The framework has three layers, each answering one research question.

| Layer | Research question | Method |
|---|---|---|
| 1. Vision cone | **RQ1.** How much does a 120 degree vision cone change which pass counts as the best option, compared with an omniscient model? | Facing direction from the passer's movement into the pass; teammates inside the 120 degree cone are visible. Every hypothetical pass is valued with VAEP. |
| 2. Decision bias | **RQ2.** Among visible, onside and completable options, how often do players bypass a better one? | Offside from the freeze frame, completion probability (xP), expected threat (xT), and a risk-adjusted version using expected value. |
| 3. Spatial memory | **RQ3.** Can an LSTM locate teammates outside the cone, and does that explain passes made to them (blind passes)? | An LSTM trained on where the passer saw each teammate in the previous 5 s (PFF tracking), with a pass-arrival model to test whether a pass from memory would have worked. |

## Headline results

| | Result |
|---|---|
| **RQ1** | Restricting to the vision cone changes the best option in **57.9%** of passes (paired Cohen's d = 0.28). Keeping the same number of teammates at random, or pointing the cone in a random direction, changes it at least as often (58.5% and 59.4%), so the effect comes from how many options are kept, not the cone's direction. |
| **RQ2** | A better visible, onside, completable option is bypassed in **38.9%** of passes (95% CI 38.1 to 39.8). An omniscient model would report 74.2%. |
| RQ2 with risk | Giving every option an expected value lowers the bypass rate to **34.7%**. |
| **RQ3** | With PFF tracking, the LSTM locates unseen teammates with a mean error of **3.03 m**, 1.03 m better than assuming they have not moved. **42.3%** of blind passes go to a receiver the passer could have located to within 4.66 m from memory. |

All results come from 166 international matches (Euro 2020, Euro 2024 and the 2022 World Cup), with PFF tracking for the 64 World Cup matches.

## Terms used in the code and notebooks

- **Visible / invisible:** a teammate inside or outside the passer's 120 degree vision cone. The cone is a proxy built from the passer's movement into the pass, not a measure of gaze or perception.
- **Maximal-VAEP option:** a teammate whose zone has higher expected threat than the passer's zone, and to whom a pass is worth more (VAEP) than the pass actually made. Not called a "forward" option because about 10% of them are level with or behind the passer.
- **Legitimate option:** a maximal-VAEP option that is also visible, onside and has a completion probability of at least 0.05.
- **Bypass rate:** the share of passes with at least one legitimate option that was not taken (the RQ2 measure).
- **Blind pass:** a completed pass to a receiver outside the passer's cone.
- **Memory-feasible:** the LSTM, using only the moments the teammate was inside the passer's cone, places the teammate within tau = 4.66 m (half the median gap between neighbouring defenders in a back line).

## Repository structure

```
Code/
├── notebooks/        the analysis, numbered in reading order (see below)
├── src/              helper modules used by the notebooks
│   ├── vision_cone.py         Layer 1: facing direction and the 120 degree cone
│   ├── offside.py             Layer 2: offside line from the freeze frame
│   ├── pff.py                 PFF loading, coordinates and linking to StatsBomb
│   ├── pff_memory.py          RQ3: the passer's cone and seen history from tracking
│   ├── reach.py               pass-arrival model from tracking
│   ├── demo_export.py         exports World Cup tracking and passes for the demo
│   └── demo_export_freeze.py  exports the freeze-frame counterfactuals for the demo
├── demo/
│   └── pass_explorer.html     the interactive demo
├── data/             not in git (see Data below)
└── requirements.txt
```

## Notebooks

Run them in order. Each notebook starts with a short explanation and ends with a summary of its results.

| | Notebook | What it does |
|---|---|---|
| 01 | `01_layer1_visual_constraint` | Facing direction, the vision cone, VAEP training and RQ1 |
| 02 | `02_layer2_bias_analysis` | Offside, xP, maximal-VAEP options and the RQ2 bypass rate |
| 03 | `03_layer2_risk_and_context` | RQ2 after risk, and how game state, pressure, position and pitch zone affect it |
| 04 | `04_layer3_lstm_sequences` | Linking StatsBomb freeze frames into tracks for the LSTM |
| 05 | `05_layer3_lstm_training` | Training the StatsBomb LSTM and comparing it with simple baselines |
| 06 | `06_layer3_blind_passes` | The StatsBomb blind-pass test, and the coordinate checks |
| 07 | `07_layer3_memory_feasibility` | RQ2 split into visible, memory-feasible and memory-infeasible teammates |
| 08 | `08_layer3_pff_link` | Linking PFF tracking to the StatsBomb passes |
| 09 | `09_layer3_pff_lstm_check` | The StatsBomb LSTM judged against PFF positions |
| 10 | `10_layer3_pff_reachability` | Would a pass to a given spot actually arrive? |
| 11 | `11_layer3_pff_rq3_memory` | RQ3 part 1: which teammates were inside the passer's cone in the 5 s before each pass |
| 12 | `12_layer3_pff_rq3_lstm` | RQ3 part 2: the LSTM spatial memory model |
| 13 | `13_layer3_pff_rq3_answer` | RQ3 part 3: how many blind passes are memory-feasible |
| 14 | `14_layer3_pff_facing_check` | Checking the facing direction and cone width behind RQ1 and RQ2 |
| 15 | `15_layer3_pff_better_options` | RQ2 against PFF analysts' better-option labels, and the two poster scenarios |
| 16 | `16_rq1_random_removal_check` | RQ1 against two random baselines: teammates kept at random, and the cone pointed in a random direction |

Figures are saved to `data/results/figures/`, and intermediate results to `data/results/`.

## Data

The `data/` folder is not in the repository because of its size (about 9 GB). It is rebuilt by running the notebooks.

- **StatsBomb open data** (events and 360 freeze frames). Downloaded automatically by `statsbombpy` the first time a match is used, and cached in `data/statsbomb_cache/`. Competitions used: UEFA Euro 2024 (competition 55, season 282), UEFA Euro 2020 (55, 43) and FIFA World Cup 2022 (43, 106).
- **PFF FC World Cup 2022 data** (tracking, events, rosters and metadata), freely available from PFF FC. Download it and extract it into `data/pff/` so that the folders look like this:

  ```
  data/pff/Metadata*/Metadata/{game_id}.json
  data/pff/Rosters*/Rosters/{game_id}.json
  data/pff/Event Data*/Event Data/{game_id}.json
  data/pff/Tracking Data*/Tracking Data/{game_id}.jsonl.bz2
  ```

  The folder names can carry a download timestamp; `src/pff.py` finds them with a wildcard.

## Setup

The project uses **Python 3.11** (not 3.13). On Windows:

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebooks in VS Code or Jupyter and select the `venv` kernel. Use the same kernel for every notebook: results are saved as pickle files, and a file saved with a different numpy version may not load in another environment.

Two library versions are pinned on purpose, because `socceraction` needs them: `pandera==0.17.2` and `pydantic<2`. No GPU is needed; the LSTMs train on CPU.

## The demo

The published version is linked at the top of this file and works in any up-to-date browser, including on a phone.

To run it locally, start a local web server from the `Code` folder (for example VS Code's Live Server extension) and open `demo/pass_explorer.html`. It reads its data from `data/demo/`, which is written by:

```
cd notebooks
python ../src/demo_export_freeze.py
python ../src/demo_export.py
```

Opening the HTML file directly from disk (without a server) will not work, because browsers block it from reading the data files.

## Limitations

- StatsBomb 360 has no body-orientation field, so the cone follows the passer's direction of movement, not where they were looking. A check against PFF tracking shows the aggregate results do not depend on this.
- Freeze frames and broadcast tracking only show players on camera, and positions from the two providers differ by about 3 m.
- Hypothetical passes are valued at the teammate's current position, so through balls into space ahead of a run are not represented.
- Only three international tournaments are covered, and tracking only for the 2022 World Cup.

## Acknowledgements

Data from [StatsBomb Open Data](https://github.com/statsbomb/open-data) and PFF FC. Action values from [socceraction](https://github.com/ML-KULeuven/socceraction) (VAEP). Expected threat grid from Karun Singh.
