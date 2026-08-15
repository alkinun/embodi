# 023: deterministic expert transfer

## question

does xperience action-expert transfer still improve robot control under strict
deterministic training and matched checkpoint selection?

## setup

- compare expert-only transfer against a fully fresh robot core.
- use the center-weighted dataset, model seed 3993, loader seed 3992, strict
  determinism, and the experiment 020 training protocol.
- reuse transferred checkpoints at steps 700 through 1,300.
- train one scratch run and save the same 100-step checkpoint grid.
- screen both conditions on ten paired scenes per range with seed 23000.
- select the best checkpoint within each condition by total success over 30
  scenes; ties prefer the earlier checkpoint.
- confirm both selected checkpoints on 50 paired scenes per range with seed
  23500, deterministic ik, horizon 16, and a 500-step limit.

the primary endpoint is paired total success. range-level success and lift are
secondary. both conditions receive identical checkpoint-selection freedom.

## result

pending.

## finding

pending.

## decision

- retain expert transfer only if it wins the paired deterministic comparison.
