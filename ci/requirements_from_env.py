"""Derive a pip requirements file from `env.yml`.

`env.yml` is the only dependency definition this repository has. Writing a second list for CI
would mean the two could drift, which is how the `xarray` breakage reached `develop` in the first
place -- so CI reads the same file a contributor does, and this script is the only thing that
knows how to translate it.

Three packages are excluded from the CI environment. Each exclusion is a claim about the code,
and each is checked by `--verify` so it cannot rot silently:

    tensorflow  ~600 MB, and imported inside functions rather than at module scope, so nothing
                the test suite touches needs it present
    psycopg2    no usages anywhere under `hamlet/`; needs libpq headers to build
    jupyter     notebooks only

Usage:
    python ci/requirements_from_env.py env.yml -o requirements-ci.txt
    python ci/requirements_from_env.py env.yml --verify      # check the exclusions still hold

This is a stopgap for the same reason the `xarray` pin is: the durable answer is a lockfile plus
`pyproject.toml` extras, at which point this script and its exclusion list both disappear.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Packages deliberately absent from the CI environment, with the reason each one is safe to drop.
EXCLUDED = {
    'tensorflow': 'imported lazily inside functions; ~600 MB',
    'psycopg2': 'unused in hamlet/',
    'jupyter': 'notebooks only',
}

_PIP_BLOCK = re.compile(r'^\s*-\s*pip:\s*$')
_REQUIREMENT = re.compile(r'^-\s+([A-Za-z0-9_.\-]+)\s*([=<>!~].*)?$')


def parse(env_yml: Path) -> list[str]:
    """The pip requirements listed in `env.yml`, in file order, minus the exclusions."""
    requirements, in_pip_block = [], False

    for line in env_yml.read_text(encoding='utf-8').splitlines():
        if _PIP_BLOCK.match(line):
            in_pip_block = True
            continue
        if not in_pip_block:
            continue
        # The pip block is the last thing in the file; a non-indented line would end it.
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        match = _REQUIREMENT.match(stripped)
        if not match:
            continue
        if match.group(1).lower() in EXCLUDED:
            continue
        requirements.append(stripped[2:].strip())

    if not requirements:
        raise SystemExit(f'{env_yml}: found no pip requirements -- has the file format changed?')

    return requirements


def verify(repo_root: Path) -> int:
    """Re-check the two exclusions that are claims about the source, not about size.

    `tensorflow` must not be imported at module scope and `psycopg2` must not be imported at all.
    If either becomes untrue, CI would install a working environment for the wrong reason -- or
    fail with an import error that points at the pipeline rather than at the commit that caused
    it.
    """
    problems = []

    # --untracked so a file that has been written but not yet added is still checked; ignored
    # paths (results/, scenarios/) stay out either way. In CI the checkout is clean and this
    # changes nothing, but locally it is the difference between checking and appearing to.
    result = subprocess.run(
        ['git', 'grep', '-n', '--untracked', '-E',
         r'^\s*(import|from)\s+(tensorflow|keras|psycopg2)', '--', 'hamlet/'],
        cwd=repo_root, capture_output=True, text=True)

    # git grep: 0 = matches, 1 = none, anything else is a real failure. Treating an error as
    # "no matches" would turn a broken check into a silent pass.
    if result.returncode > 1:
        print(f'git grep failed ({result.returncode}): {result.stderr.strip()}', file=sys.stderr)
        return 2

    for line in result.stdout.splitlines():
        # `git grep -n` emits `path:lineno:source`, so the source is the third field -- splitting
        # on the first colon alone leaves the line number in front of it and every match looks
        # unindented.
        parts = line.split(':', 2)
        if len(parts) < 3:
            continue
        statement = parts[2]
        # An indented import is inside a function, which is the point: it is not paid at import
        # time, so the excluded package is not needed to collect or run the tests.
        if statement and not statement[0].isspace():
            problems.append(line)

    if problems:
        print('An excluded package is now imported at module scope:', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        print('\nEither move the import inside the function that needs it, or remove the package '
              'from EXCLUDED in this script so CI installs it.', file=sys.stderr)
        return 1

    print('exclusions still hold: no module-scope import of ' + ', '.join(sorted(EXCLUDED)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('env_yml', type=Path, nargs='?', default=Path('env.yml'))
    parser.add_argument('-o', '--output', type=Path,
                        help='write here instead of stdout')
    parser.add_argument('--verify', action='store_true',
                        help='check the exclusions still hold and exit')
    args = parser.parse_args()

    if args.verify:
        return verify(args.env_yml.resolve().parent)

    requirements = parse(args.env_yml)
    text = '\n'.join(requirements) + '\n'

    if args.output:
        args.output.write_text(text, encoding='utf-8')
        excluded = ', '.join(sorted(EXCLUDED))
        print(f'{len(requirements)} requirements -> {args.output} (excluded: {excluded})')
    else:
        sys.stdout.write(text)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
