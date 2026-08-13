# 006: action-expert scaling

## question

which expert size works best under a fixed short budget?

## setup

- one screening seed.
- frozen vlm and 100 fixed xperience clips.
- 1,000 human updates.
- 1,500 so-101 updates on 100 demonstrations.
- matched scratch and transfer runs.
- 30 paired simulator scenes with deterministic ik.

| size | width | layers | heads | parameters |
| --- | ---: | ---: | ---: | ---: |
| small | 256 | 8 | 8 | 9.24M |
| medium | 512 | 12 | 8 | 53.12M |
| large | 640 | 16 | 10 | 109.05M |

## result

| size | human validation loss | robot scratch loss | robot transfer loss | scratch success | transfer success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9.24M | `0.3347` | `0.001021` | `0.000661` | 4/30 (13%) | 11/30 (37%) |
| 53.12M | `0.2127` | `0.002908` | `0.000486` | 6/30 (20%) | 19/30 (63%) |
| 109.05M | `0.2081` | `0.001358` | `0.000477` | 3/30 (10%) | 7/30 (23%) |

pretraining helped every size. human loss changed little from 53M to 109M. the
109M model was under-trained and weak in control.

## finding

the 53M expert has the best transfer and compute tradeoff in this screen.

## decision

keep `width=512`, `layers=12`, and `heads=8`. this is not a scaling law. do not
scale the model before scaling data and training.
