"""Command-line build:  python -m livery_maker.cli project.json [--zip]"""
import argparse
import sys
import time

from .engine.common import Project
from .engine.export import build_all, validate, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description='Build every livery in a Livery Maker project file.')
    ap.add_argument('project')
    ap.add_argument('--zip', action='store_true', help='also zip each livery folder')
    a = ap.parse_args(argv)
    with open(a.project) as fh:
        proj = Project.from_json(fh.read())
    probs = validate(proj)
    if probs:
        print('Cannot build:\n  ' + '\n  '.join(probs))
        return 1
    for w in warnings(proj):
        print('Warning:', w)
    t0 = time.time()
    folders = build_all(proj, progress=lambda m: print(m, flush=True), make_zips=a.zip)
    print(f'Built {len(folders)} liveries in {time.time() - t0:.0f}s -> {proj.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
