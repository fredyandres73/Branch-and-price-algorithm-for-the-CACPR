# Branch-and-price for the C-ACPR

Source code, instances and computational results of the paper

> **A branch-and-price algorithm for the capacitated angular set covering problem with rotation**
> F. Barriga-Gallegos, J. Montecinos, A. Lüer-Villagra, G. Gutiérrez-Jarpa. Submitted to *Computers & Operations Research*.

The **Capacitated Angular Set Covering Problem with Rotation (C-ACPR)** locates facilities and installs at most `Q`
directional servers at each of them, choosing for every server its configuration (coverage angle), position
(orientation), type (reach) and rotation, so that every demand point is covered at minimum total cost.


## Instance format

Whitespace-separated tokens, read in this order:

| Tokens | Meaning |
| --- | --- |
| `P U C S` | demand points, candidate locations, configurations, server types |
| `conf[C]` | coverage angle of each configuration, in degrees |
| `pos[C]` | number of positions of each configuration, `360 / conf` |
| `covd[S]` | covering distance of each server type at 90 degrees |
| `g` | cost of locating a facility, identical for every location |
| `srv[S][C]` | cost of installing a server of each type in each configuration |
| `P` pairs | coordinates of the demand points |
| `U` pairs | coordinates of the candidate locations |

File names follow `dataset_<demand points>P_<candidates>U_<server types>S.txt`. The twelve Valparaíso instances add
`_<configurations>C`.