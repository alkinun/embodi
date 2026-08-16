# data research

the generalist objective requires measuring independent behavioral breadth, not
only frame or clip count. adjacent Xperience anchors share most of their
1.067-second action horizon, so they are correlated views of the same behavior.

## data layers

1. broad egocentric video and language should teach visual semantics, objects,
   affordances, temporal context, and task structure.
2. human motion data with synchronized 3d hands or body should teach canonical
   action priors independent of a robot embodiment.
3. multi-robot trajectories should connect canonical intent to controllable
   states, recovery behavior, and embodiment-specific actions.
4. target-robot demonstrations should calibrate the final state and action path.

the current Xperience pipeline addresses only the second layer and only for the
right hand. the VLM is frozen during this stage, so adding generic videos without
motion labels will not improve the current action objective by itself. broad
video-language pretraining needs a separate temporal or semantic objective, and
its contribution must be tested by downstream transfer rather than assumed.

## required accounting

every pretraining dataset and mixture should report:

- immutable source revision and file hashes;
- participants, sessions, episodes, main tasks, subtasks, objects, and scenes;
- union labeled seconds and non-overlapping action-window capacity;
- trajectory and motion-bin coverage, not only clip count;
- train/validation/test overlap by participant, session, task, prompt, and scene;
- sampling weights and presentations per source;
- label validity, rejection reasons, and missing-modality rates.

training writes `metrics.jsonl` locally so validation evidence does not depend
on terminal capture or an external tracking service.
`scripts/evaluate_xperience.py` evaluates a saved checkpoint on the complete
frozen validation split and reports each held-out session separately.

`scripts/prepare_xperience_split.py` freezes fixed episode-disjoint selections
before training. `scripts/analyze_xperience_data.py` provides the currently
available session, task, prompt, motion, and temporal-overlap accounting.

the post-hoc experiment 009 audit shows why this matters. increasing from 1,000
to 10,000 selected clips increased union labeled time only from 584.8 to 964.6
seconds and non-overlapping-window capacity only from 431 to 796. the nominal
10x clip increase therefore added 1.65x temporal coverage and 1.85x
non-overlapping capacity while leaving the same five main tasks.

## evaluation ladder

1. held-out human sessions and tasks with locked split manifests;
2. few-shot transfer to held-out simulated tasks, objects, goals, and visual
   domains;
3. transfer across robot embodiments at matched data and compute;
4. end-to-end control without oracle canonical state or simulator ik;
5. physical evaluation only after calibration and safety gates pass.

model-size claims require at least three training seeds, matched presentations
and compute, fixed checkpoint rules, exact parameter counts, and an untouched
final benchmark. the current 53.12M expert is the best internal choice at the
tested short budget; it is not yet established as globally optimal for its size.

## near-term sequence

1. execute experiment 034 to test breadth at fixed volume.
2. expand the cache only after cataloging candidate sessions by task and label
   validity; prioritize new tasks, participants, scenes, and objects over nearby
   windows.
3. create a task-disjoint simulation benchmark before using it for model
   selection, retaining an untouched final test split.
4. compare data mixtures and expert sizes under matched compute.
5. add broad egocentric video objectives only with modality ablations that show
   improved held-out task transfer.
