"""Benchmark: liquid benzene at 290 K on LET, four incident energies fitted at once.

    python examples/benzene_let.py                  # best fit, about 5 minutes
    python examples/benzene_let.py --sample 3000    # plus posterior sampling
    python examples/benzene_let.py --holdout        # plus the leave-one-energy-out test

Expected best fit (jump diffusion + isotropic rotation): D ≈ 0.172 Å²/ps,
τ₀ ≈ 0.95 ps, D_r ≈ 0.166 ps⁻¹; PFG-NMR gives D ≈ 0.18 Å²/ps.
"""

import argparse
import time
from pathlib import Path

import uqens

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with the benzene .nxspe files")
parser.add_argument("--model", default="jump+iso", choices=sorted(uqens.LADDER), help="model from the ladder")
parser.add_argument("--random-starts", type=int, default=3, help="random starts on top of the 3 seeded ones")
parser.add_argument("--sample", type=int, default=0, metavar="STEPS", help="run the sampler for this many steps")
parser.add_argument("--holdout", action="store_true", help="leave-one-energy-out prediction test")
args = parser.parse_args()

files = [args.data / f"benzene_290_{e}_inc.nxspe" for e in ("124", "197", "360", "861")]
data = uqens.let.load(files, verbose=True)
problem = uqens.Problem(uqens.LADDER[args.model], data)

t = time.time()
best = uqens.fit(problem, n_random=args.random_starts, verbose=True)
print(f"\n{best.summary()}\n({time.time() - t:.0f} s)")

if args.sample:
    t = time.time()
    post = uqens.sample(problem, best.x, n_steps=args.sample)
    print(f"\n{post.summary()}\n({time.time() - t:.0f} s)")

if args.holdout:
    print("\nleave-one-energy-out:")
    rows = uqens.validate.holdout(problem.spec, data, problem, best.x, verbose=True)
    print(f"held-out chi2, all energies: {sum(r['chi2_transfer'] for r in rows):.1f}")
