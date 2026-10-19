# Rescue Rover Scoring

Mission completion has priority over efficiency. A successful mission receives
a 10,000-point bonus. With the 100-action limit, every successful run has a
higher score than every unsuccessful run.

| Action or outcome | Score change |
|---|---:|
| Move one tile | -3 |
| Additional cost when entering a stable tile with traction enabled | -3 |
| Additional cost when entering a slippery tile with traction enabled | 0 |
| Structural scan | -2 |
| Surface scan | -2 |
| Enable traction | -1 |
| Disable traction | -1 |
| Retrieve module | -1 |
| Valid stop | 0 |
| Invalid action | -5 |
| Stop before carrying the module at the exit | -100 |
| Reach the action limit | -100 |
| Enter a slippery tile without traction | -100 |
| Enter a fragile tile | -1,000 |
| Stop while carrying the module at the exit | +10,000 |

Action costs and outcome scores are additive. For example, enabling traction,
crossing one slippery tile, and disabling traction costs 5 points
(`-1 - 3 - 1 = -5`). A three-move stable detour costs 9 points. If traction is
already enabled, crossing one stable tile costs 6 points. Disabling traction,
crossing that tile, and enabling traction again costs 5 points
(`-1 - 3 - 1 = -5`).

Entering a fragile tile changes the score by `-3 - 1000 = -1003`. No terrain
efficiency surcharge applies because the move does not enter a stable tile.

Entering a slippery tile without traction changes the score by
`-3 - 100 = -103`. The rover moves onto the tile, the outcome event is
`slipped`, and the episode continues. Entering a fragile tile ends the episode.
A failed scan shape, non-adjacent move, redundant traction toggle, or invalid
retrieval leaves the environment state unchanged and applies the invalid-action
score.

The reported efficiency measures are action count, move count, scan count,
traction-enabled move count, traction-toggle count, unprotected-slip count, and
invalid-action count. "Traction use" in this starter means the combination of
traction-enabled moves and toggle actions. Scores provide one summary measure;
reports should retain the component measures so different agent behaviors
remain interpretable.
