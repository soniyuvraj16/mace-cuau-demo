"""Write one VASP calculation directory per structure in data/structures.xyz.

  vasp/calcs/<name>/POSCAR KPOINTS INCAR [POTCAR] meta.json
  vasp/calcs/list.txt           one directory name per line, for a job array

Atoms are reordered Cu-first for the POSCAR; meta.json keeps the permutation
so collect.py can map forces back to the structure's own site order.

POTCARs are concatenated from --potcar-dir if given (expects
<dir>/Cu/POTCAR and <dir>/Au/POTCAR, the potpaw_PBE layout). They are licensed,
so they stay out of git; copy them in on the cluster if you skip this here.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA  # noqa: E402

KGRID = {4: (8, 8, 8), 8: (4, 8, 8)}
SPECIES_ORDER = ("Cu", "Au")


def find_potcar(potcar_dir, element):
    for candidate in (potcar_dir / element / "POTCAR", potcar_dir / f"POTCAR.{element}", potcar_dir / f"{element}.POTCAR"):
        if candidate.exists():
            return candidate
    sys.exit(f"no POTCAR for {element} under {potcar_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "vasp" / "calcs"))
    p.add_argument("--potcar-dir", default=None, help="directory holding Cu/POTCAR and Au/POTCAR (e.g. the repo root)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    potcar_dir = Path(args.potcar_dir) if args.potcar_dir else None

    frames = read(DATA / "structures.xyz", ":")
    names = []
    for at in frames:
        name = at.info["name"]
        d = out / name
        d.mkdir(exist_ok=True)
        rank = [SPECIES_ORDER.index(s) for s in at.get_chemical_symbols()]
        perm = np.argsort(rank, kind="stable")
        ordered = at[perm]
        write(d / "POSCAR", ordered, format="vasp", direct=True, sort=False)
        kx, ky, kz = KGRID[len(at)]
        (d / "KPOINTS").write_text(f"{name}\n0\nGamma\n{kx} {ky} {kz}\n0 0 0\n")
        shutil.copy(ROOT / "tools" / "vasp" / "INCAR", d / "INCAR")
        present = [s for s in SPECIES_ORDER if s in ordered.get_chemical_symbols()]
        if potcar_dir:
            with open(d / "POTCAR", "wb") as f:
                for el in present:
                    f.write(find_potcar(potcar_dir, el).read_bytes())
        (d / "meta.json").write_text(json.dumps({"name": name, "perm": perm.tolist(), "potcar_order": present}))
        names.append(name)

    (out / "list.txt").write_text("\n".join(names) + "\n")
    bundle = out.parent
    template = (ROOT / "tools" / "vasp" / "submit_array.sh").read_text()
    (bundle / "submit_array.sh").write_text(template.replace("--array=1-180%30", f"--array=1-{len(names)}%30"), newline="\n")
    shutil.copy(ROOT / "HANDOFF_GRACE.md", bundle / "HANDOFF_GRACE.md")
    shutil.copy(ROOT / "tools" / "vasp" / "TOPUP.md", bundle / "TOPUP.md")
    (bundle / "tools").mkdir(exist_ok=True)
    shutil.copy(ROOT / "tools" / "vasp" / "collect.py", bundle / "tools" / "collect.py")
    shutil.copy(ROOT / "scripts" / "common.py", bundle / "tools" / "common.py")
    (bundle / "data").mkdir(exist_ok=True)
    shutil.copy(DATA / "structures.xyz", bundle / "data" / "structures.xyz")
    print(f"wrote {len(names)} calculation directories under {out}")
    print(f"bundle {bundle} is self-contained: submit_array.sh, HANDOFF_GRACE.md, tools/collect.py, data/structures.xyz")
    print(f"POTCAR {'included' if potcar_dir else 'NOT included (pass --potcar-dir or add on the cluster)'}")


if __name__ == "__main__":
    main()
