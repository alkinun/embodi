# 003: control and decoder

## question

why did accurate canonical predictions fail in control?

## setup

compare native control, deterministic ik, and learned decoders.

## result

- native controller: 20/20 success.
- canonical commands through deterministic ik: 20/20.
- original learned decoder: 0/20.
- residual decoder on 100 demonstrations: 15/20.
- stale 2-degree action limits reduced the ik teacher to 0/20.

policy-state decoder training raised baseline control to 62% and 56%.

## finding

the decoder and stale action limits caused the control failure.

## decision

- disable stale action limits.
- train decoders on policy states.
- select decoders with closed-loop success, not mse.
