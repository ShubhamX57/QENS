"""Gate 1: why does the published LET analysis give D = 0.182 Å²/ps?

Richardson et al. 2026 (arXiv:2603.06080) fitted the 1.97 and 3.60 meV data over
±1.25 meV, Q 0.59–1.53 and 0.67–1.72 Å⁻¹, after subtracting the empty can, with
Fickian translation, anisotropic rotation to l = 2, one scale per energy times
exp(−Q²⟨u²⟩/3), a flat background per Q, no can term, no detailed balance, and the
sample frozen at 260 K as the resolution.

Part A, both energies: start from uqens's choices and switch them to the paper's,
one at a time and each on its own.  This repository has no empty-can or 260 K run
at 1.97 meV, so the can cannot be subtracted and 1.97 meV keeps the analytic
resolution.

Part B, 3.60 meV only, where the empty can and the 260 K run exist: the paper's
full set of choices, then each one switched back.

    python examples/gate1_published_setup.py            # about 15 minutes
    python examples/gate1_published_setup.py --part B   # about 3 minutes
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import uqens
from uqens import ModelSpec, let

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", type=Path, default=ROOT / "data")
parser.add_argument("--part", choices=["A", "B", "both"], default="both")
parser.add_argument("--random-starts", type=int, default=2)
parser.add_argument("--out", type=Path, default=Path("gate1_results.json"))
args = parser.parse_args()

PAPER_Q = {
    "qmin": {1.97: 0.59, 3.60: 0.67},
    "qmax": {1.97: 1.53, 3.60: 1.72},
    "dq_target": {1.97: 0.073, 3.60: 0.096},
    "window": 1.25,
}
CAN = args.data / "empty_360_inc.nxspe"
FROZEN = args.data / "benzene_260_360_inc.nxspe"
OURS = ModelSpec("uqens choices", "fickian", "aniso", lmax=2)
PAPER = {"amplitudes": "global", "use_el": False, "detailed_balance": False, "resolution": "measured"}
rows = []


def run(part, label, spec, data):
    t = time.time()
    best = uqens.fit(uqens.Problem(spec, data), n_random=args.random_starts)
    p = best.physics()
    chi = best.chi2()
    per_n = ", ".join(f"{chi[d.ei] / len(d.y):.2f}" for d in data)
    others = "; ".join(
        f"+{v - best.neglogpost:.0f}: D {best.problem.physics(x)['D']:.3f}, Ds/Dt "
        f"{best.problem.physics(x)['Ds'] / best.problem.physics(x)['Dt']:.2f}"
        for v, x in best.modes[1:]
    )
    rows.append({"part": part, "configuration": label, **p, "neglogpost": best.neglogpost, "chi2_per_n": per_n})
    print(
        f"{label:48s} {p['D']:6.3f} {p['Dt']:6.3f} {p['Ds']:6.3f} {p['Ds'] / p['Dt']:6.2f}  {per_n:11s}"
        f" ({time.time() - t:3.0f} s){'  other optima: ' + others if others else ''}",
        flush=True,
    )


header = f"{'configuration':48s} {'D':>6s} {'Dt':>6s} {'Ds':>6s} {'Ds/Dt':>6s}  chi2/N"

if args.part in ("A", "both"):
    data = let.load([args.data / "benzene_290_197_inc.nxspe", args.data / "benzene_290_360_inc.nxspe"], **PAPER_Q)
    let.attach_resolution(data[1], FROZEN)
    steps = [
        ("one scale per energy x exp(-Q2<u2>/3)", {"amplitudes": "global"}),
        ("no can term", {"use_el": False}),
        ("no detailed balance", {"detailed_balance": False}),
        ("260 K sample as resolution (3.60 meV)", {"resolution": "measured"}),
    ]
    print(f"\nPart A: 1.97 + 3.60 meV, files as they are (can not subtracted)\n{header}")
    run("A", "uqens choices", OURS, data)
    cumulative = {}
    for label, change in steps:
        cumulative.update(change)
        run("A", f"+ {label}", dataclasses.replace(OURS, **cumulative), data)
    for label, change in steps:
        run("A", f"only: {label}", dataclasses.replace(OURS, **change), data)

if args.part in ("B", "both"):
    f360 = [args.data / "benzene_290_360_inc.nxspe"]
    raw = let.load(f360, **PAPER_Q)
    sub = let.load(f360, background={3.60: CAN}, **PAPER_Q)
    let.attach_resolution(sub[0], FROZEN, background=CAN)
    paper = dataclasses.replace(OURS, **PAPER)
    print(f"\nPart B: 3.60 meV only, empty can subtracted where stated\n{header}")
    run("B", "uqens choices, can not subtracted", OURS, raw)
    run("B", "uqens choices, can subtracted", OURS, sub)
    run("B", "paper's choices", paper, sub)
    run("B", "paper's, but analytic resolution", dataclasses.replace(paper, resolution="analytic"), sub)
    run("B", "paper's, but a free scale per Q-bin", dataclasses.replace(paper, amplitudes="per_bin"), sub)
    run("B", "paper's, but rotation to the sum rule", dataclasses.replace(paper, lmax=None), sub)
    run("B", "paper's, but jump diffusion", dataclasses.replace(paper, translation="jump"), sub)

args.out.write_text(json.dumps(rows, indent=2))
print(f"\nPublished: D = 0.182 ± 0.002 Å²/ps, Ds/Dt ≈ 11.  Results saved to {args.out}")
