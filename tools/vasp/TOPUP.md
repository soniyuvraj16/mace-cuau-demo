# Top-up: 6 more calculations

After the first 180 runs were done, a bug in the symmetry reduction turned out
to have merged two pairs of genuinely different 8-atom orderings, so two
orderings (`00010101` and `01010111`) were missing. This bundle adds them:
six new directories (`180_…` to `185_…`), an updated `calcs/list.txt` (186
lines), an updated `submit_array.sh`, and an updated `data/structures.xyz`.
Nothing about the existing 180 directories changes.

On Grace, in the directory that contains `vasp/`:

```
tar xzf cuau_vasp_topup.tar.gz        # adds files into the existing vasp/
cd vasp
sbatch --array=181-186 submit_array.sh
```

When the six have converged:

```
python tools/collect.py                # -> data/train.xyz (135), data/holdout.xyz (51)
```

and send back `data/train.xyz` and `data/holdout.xyz` as before. Settings are
identical to the first 180; see HANDOFF_GRACE.md for everything else.
