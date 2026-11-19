# Structural check – `heart`

Parameters: `params.toml`. Material: PLA (FDM, in-plane), E = 3000 MPa, strength 45 MPa. Layers [9.0, 8.0, 9.0, 8.0, 6.0, 7.0, 8.0, 7.0, 6.0] mm. Gears module 1.5, z30/90. Rollers 684ZZ. Masses 0.25/0.2/0.08 kg. Pen drag 0.40 N.

Validation: statics chain vs inverse dynamics, max motor-torque difference 1.3e-04 N·m (peak 8.1e-02 N·m).

Each spring sits in the spacer gap under its hub, around the fixed shoulder axle: one leg in the keyed collar (D-flat), the other in the hub's leg hole. Wind it by `wind` before inserting the hub leg.

| hub | preload (N·m) | wire d (mm) | mean D (mm) | OD / ID (mm) | coils | body / gap (mm) | rate (N·mm/rad) | wind (°) | torque ± over travel | stress / allowed (MPa) | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | +0.30 | 1.4 | 17.9 | 19.3 / 16.5 | 6 | 9.8 / 11.0 | 116 | 149 | 23 % over 69° | 1457 / 1579 | yes |
| M2 | +0.22 | 1.2 | 19.7 | 20.9 / 18.5 | 7 | 9.6 / 11.0 | 49 | 259 | 12 % over 61° | 1518 / 1615 | yes |
| M3 | +0.08 | 1.1 | 19.6 | 20.7 / 18.5 | 13 | 15.4 / 16.0 | 19 | 247 | 16 % over 80° | 743 / 1635 | yes |

## Case A – cam drive (as designed, spring-preloaded)
duration 7.4 s, peak torque the arm needs [0.173, 0.132, 0.028] N·m

| member | peak axial F (N) | peak moment (N·mm) | peak stress (MPa) | strain (µε) | SF |
|---|---|---|---|---|---|
| upper arm (S-E) |  | 173 | 0.288 | 96 | 156 |
| forearm (T2-E-W) |  | 130 | 0.412 | 137 | 109 |
| hand (W-pen) |  | 28 | 0.098 | 33 | 458 |
| hand crank (W-B2) |  | 28 | 0.100 | 33 | 448 |
| M2 elbow crank (S-C2) |  | 132 | 0.267 | 89 | 169 |
| M3 wrist crank (S-A1) |  | 28 | 0.072 | 24 | 628 |
| bell-crank arm B |  | 28 | 0.079 | 26 | 569 |
| bell-crank arm A |  | 28 | 0.081 | 27 | 552 |
| coupler E (C2-T2) | 2.20 |  | 0.032 | 11 | 1417 (buckling SF 109) |
| coupler A (A1-A2) | 0.81 |  | 0.020 | 7 | 2233 (buckling SF 101) |
| coupler B (B1-B2) | 0.77 |  | 0.014 | 5 | 3123 (buckling SF 304) |

| joint | peak force (N) | bearing (MPa) | net tension ×Kt3 (MPa) | SF |
|---|---|---|---|---|
| S hub (axle 8 mm) | 2.16 | 0.033 | 0.059 | 763 |
| elbow E (upper arm lug) | 2.16 | 0.063 | 0.059 | 718 |
| elbow E (forearm lug) | 2.91 | 0.113 | 0.106 | 400 |
| wrist W | 0.90 | 0.035 | 0.042 | 1066 |
| T2 (coupler E) | 2.20 | 0.057 | 0.095 | 472 |
| A1/A2 (coupler A) | 0.81 | 0.031 | 0.060 | 744 |
| B1/B2 (coupler B) | 0.77 | 0.022 | 0.043 | 1041 |

| gear mesh | Ft (N) | root σ hub (MPa) | root σ follower (MPa) | contact (MPa) | hub-root strain (µε) | SF | load reverses? |
|---|---|---|---|---|---|---|---|
| M1 hub z30 / follower z90 (m1.5, b8) | 20.4 | 7.11 | 5.78 | 19.6 | 2371 | 6.3 | no (min 0.120 N·m) |
| M2 hub z30 / follower z90 (m1.5, b8) | 14.7 | 5.10 | 4.14 | 16.6 | 1701 | 8.8 | no (min 0.108 N·m) |
| M3 hub z30 / follower z90 (m1.5, b8) | 4.8 | 1.67 | 1.35 | 9.5 | 555 | 27.0 | no (min 0.051 N·m) |

| cam | roller force (N) | Hertz on groove (MPa) | SF | follower root σ (MPa) |
|---|---|---|---|---|
| cam M1 | 20.0 | 37.7 | 1.2 | 5.28 |
| cam M2 | 16.8 | 33.8 | 1.3 | 3.79 |
| cam M3 | 3.8 | 15.7 | 2.9 | 1.24 |

Camshaft torque (ideal, springs included): peak 0.370 N·m, RMS 0.161 N·m at 7.0 rpm. With a ~40 %-efficient worm the motor must supply ≈ 0.925 N·m ÷ worm ratio.

Pen-tip error: coupler stretch 25 µm, link/crank bending 619 µm (upper arm 283, forearm 446, hand 6, cranks 78), backlash + groove slop 0.00 mm (hub slop 0.00°, 0.00°, 0.00°).

## Case B – fast motors (no cams, no springs; reference)
duration 7.3 s, peak torque the arm needs [0.251, 0.132, 0.03] N·m

| member | peak axial F (N) | peak moment (N·mm) | peak stress (MPa) | strain (µε) | SF |
|---|---|---|---|---|---|
| upper arm (S-E) |  | 251 | 0.412 | 137 | 109 |
| forearm (T2-E-W) |  | 130 | 0.412 | 137 | 109 |
| hand (W-pen) |  | 30 | 0.103 | 34 | 435 |
| hand crank (W-B2) |  | 30 | 0.105 | 35 | 430 |
| M2 elbow crank (S-C2) |  | 132 | 0.277 | 92 | 163 |
| M3 wrist crank (S-A1) |  | 30 | 0.075 | 25 | 602 |
| bell-crank arm B |  | 30 | 0.082 | 27 | 546 |
| bell-crank arm A |  | 30 | 0.085 | 28 | 527 |
| coupler E (C2-T2) | 2.43 |  | 0.035 | 12 | 1283 (buckling SF 99) |
| coupler A (A1-A2) | 0.88 |  | 0.022 | 7 | 2059 (buckling SF 88) |
| coupler B (B1-B2) | 0.77 |  | 0.014 | 5 | 3132 (buckling SF 305) |

| joint | peak force (N) | bearing (MPa) | net tension ×Kt3 (MPa) | SF |
|---|---|---|---|---|
| S hub (axle 8 mm) | 2.09 | 0.032 | 0.057 | 790 |
| elbow E (upper arm lug) | 2.08 | 0.061 | 0.057 | 744 |
| elbow E (forearm lug) | 3.29 | 0.128 | 0.120 | 352 |
| wrist W | 0.96 | 0.037 | 0.045 | 1001 |
| T2 (coupler E) | 2.43 | 0.063 | 0.105 | 428 |
| A1/A2 (coupler A) | 0.88 | 0.034 | 0.066 | 686 |
| B1/B2 (coupler B) | 0.77 | 0.022 | 0.043 | 1044 |

## Case C – 5 N push on the pen tip (any direction, whole paper, springs on)

peak torque at the hubs [1.5, 1.25, 0.35] N·m

| member | peak axial F (N) | peak moment (N·mm) | peak stress (MPa) | strain (µε) | SF |
|---|---|---|---|---|---|
| upper arm (S-E) |  | 1500 | 2.512 | 837 | 18 |
| forearm (T2-E-W) |  | 1236 | 3.961 | 1320 | 11 |
| hand (W-pen) |  | 350 | 1.197 | 399 | 38 |
| hand crank (W-B2) |  | 350 | 1.237 | 412 | 36 |
| M2 elbow crank (S-C2) |  | 1250 | 2.686 | 895 | 17 |
| M3 wrist crank (S-A1) |  | 350 | 0.962 | 321 | 47 |
| bell-crank arm B |  | 350 | 0.975 | 325 | 46 |
| bell-crank arm A |  | 350 | 1.072 | 357 | 42 |
| coupler E (C2-T2) | 26.12 |  | 0.377 | 126 | 119 (buckling SF 9) |
| coupler A (A1-A2) | 15.01 |  | 0.373 | 124 | 121 (buckling SF 4) |
| coupler B (B1-B2) | 9.42 |  | 0.176 | 59 | 256 (buckling SF 24) |

| joint | peak force (N) | bearing (MPa) | net tension ×Kt3 (MPa) | SF |
|---|---|---|---|---|
| S hub (axle 8 mm) | 19.14 | 0.290 | 0.522 | 86 |
| elbow E (upper arm lug) | 19.14 | 0.556 | 0.524 | 81 |
| elbow E (forearm lug) | 35.67 | 1.382 | 1.302 | 33 |
| wrist W | 11.07 | 0.429 | 0.517 | 87 |
| T2 (coupler E) | 26.12 | 0.675 | 1.131 | 40 |
| A1/A2 (coupler A) | 15.01 | 0.582 | 1.120 | 40 |
| B1/B2 (coupler B) | 9.42 | 0.274 | 0.527 | 85 |

| gear mesh | Ft (N) | root σ hub (MPa) | SF | contact (MPa) |
|---|---|---|---|---|
| M1 hub z30 / follower z90 (m1.5, b8) | 83.1 | 28.9 | 1.6 | 40 |
| M2 hub z30 / follower z90 (m1.5, b8) | 66.4 | 23.1 | 1.9 | 35 |
| M3 hub z30 / follower z90 (m1.5, b8) | 19.7 | 6.8 | 6.6 | 19 |

Pen-tip deflection under the push: coupler stretch 385 µm, link/crank bending 7052 µm (upper arm 2501, forearm 4334, hand 73, cranks 839). A push larger than the spring preload reverses the gear load: the arm then moves through the backlash and springs back when released.
