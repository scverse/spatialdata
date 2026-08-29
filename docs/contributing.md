# Contributing guide

This document aims at summarizing the most important information for getting you started on contributing to this project.
We assume that you are already familiar with git and with making pull requests on GitHub.

For more extensive tutorials, that also cover the absolute basics,
please refer to other resources such as the [pyopensci tutorials][],
the [scientific Python tutorials][], or the [scanpy developer guide][].

[pyopensci tutorials]: https://www.pyopensci.org/learn.html
[scientific Python tutorials]: https://learn.scientific-python.org/development/tutorials/
[scanpy developer guide]: https://scanpy.scverse.org/page/dev/

:::{tip} The *hatch* project manager

We highly recommend to familiarize yourself with [`hatch`][hatch].
Hatch is a Python project manager that

- manages virtual environments, separately for development, testing and building the documentation.
  Separating the environments is useful to avoid dependency conflicts.
- allows to run tests locally in different environments (e.g. different python versions)
- allows to run tasks defined in `pyproject.toml`, e.g. to build documentation.

While the project is setup with `hatch` in mind,
it is still possible to use different tools to manage dependencies, such as `uv` or `pip`.

:::

[hatch]: https://hatch.pypa.io/latest/

## Installing dev dependencies

In addition to the packages needed to _use_ this package,
you need additional python packages to [run tests](#writing-tests) and [build the documentation](#docs-building).

:::::{tab-set}
::::{tab-item} Hatch
:sync: hatch

On the command line, you typically interact with hatch through its command line interface (CLI).
Running one of the following commands will automatically resolve the environments for testing and
building the documentation in the background:

```bash
hatch test  # defined in the table [tool.hatch.envs.hatch-test] in pyproject.toml
hatch run docs:build  # defined in the table [tool.hatch.envs.docs]
```

### VS Code

If you are using VS code, install the [hatch-code][] extension.
Additionally, make sure that the `vscode-python-environments` extension is installed (should be by default)
and `"python.useEnvironmentsExtension": true` is activated in your `settings.json`.

Next, open the "Python Environment Managers" sidebar.
You can do so by opening the command palette (Ctrl+Shift+P) and searching for `Python: Focus on Environment Managers View`.
It will show a collapsible list where you can expand "Hatch"
and activate an environment by clicking on the checkmark next to it.
As the main development environment, we recommend to use `hatch-test` with the latest supported Python version.

### Other IDEs

For other IDEs, you’ll have to point the editor at the paths to the virtual environments manually.
To get a list of all environments for your projects, run

```bash
hatch env show -i
```

This will list “Standalone” environments and a table of “Matrix” environments like the following:

```
+------------+---------+--------------------------+----------+---------------------------------+-------------+
| Name       | Type    | Envs                     | Features | Dependencies                    | Scripts     |
+------------+---------+--------------------------+----------+---------------------------------+-------------+
| hatch-test | virtual | hatch-test.py3.12-stable | dev      | coverage-enable-subprocess==1.0 | cov-combine |
|            |         | hatch-test.py3.14-stable | test     | coverage[toml]~=7.4             | cov-report  |
|            |         | hatch-test.py3.14-pre    |          | pytest-mock~=3.12               | run         |
|            |         |                          |          | pytest-randomly~=3.15           | run-cov     |
|            |         |                          |          | pytest-rerunfailures~=14.0      |             |
|            |         |                          |          | pytest-xdist[psutil]~=3.5       |             |
|            |         |                          |          | pytest~=8.1                     |             |
+------------+---------+--------------------------+----------+---------------------------------+-------------+
```

From the `Envs` column, select the environment name you want to use for development.
As the main development environment, we recommend to use `hatch-test` with the latest supported Python version.
In this example, it would be `hatch-test.py3.14-stable`.

Next, create the environment with

```bash
hatch env create hatch-test.py3.14-stable
```

Then, obtain the path to the environment using

```bash
hatch env find hatch-test.py3.14-stable
```

and manually point it to the python binary.


::::

::::{tab-item} uv
:sync: uv

A popular choice for managing virtual environments is [uv][].
The main disadvantage compared to hatch is that it supports only a single environment per project at a time,
which requires you to mix the dependencies for running tests and building docs.
This can have undesired side-effects,
such as requiring to install a lower version of a library your project depends on,
only because an outdated sphinx plugin pins an older version.

To initialize a virtual environment in the `.venv` directory of your project, simply run

```bash
uv sync --group=test --group=doc --extra=torch
```

The `.venv` directory is typically automatically discovered by IDEs such as VS Code.

::::

::::{tab-item} Pip
:sync: pip

Pip is nowadays mostly superseded by environment manager such as [hatch][].
However, for the sake of completeness, and since it’s ubiquitously available,
we describe how you can manage environments manually using `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . --group dev --group test --group doc
```

The `.venv` directory is typically automatically discovered by IDEs such as VS Code.

::::
:::::

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/
[hatch-code]: https://marketplace.visualstudio.com/items?itemName=PyPA.hatch
[uv]: https://docs.astral.sh/uv/

## Code-style

This package uses [pre-commit][]-style hooks to enforce consistent code-styles.
We recommend running them with [prek][], a fast, drop-in replacement for `pre-commit` that reads the same `.pre-commit-config.yaml`.
On every commit, the checks will either automatically fix issues with the code, or raise an error message.

To enable the checks locally, run

```bash
hatch run hatch-check-code:prek install
# or
uvx prek install
```

in the root of the repository.
prek will automatically download all dependencies when it is run for the first time.

If you didn’t run the checks locally, the `Pre-commit checks` job of the GitHub Actions CI runs them on your pull request and reports any failures.
We strongly encourage installing and running the checks locally first to understand their usage.

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [ruff][ruff-editors] and [biome][biome-editors].

[pre-commit]: https://pre-commit.com/
[prek]: https://prek.j178.dev/
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[biome-editors]: https://biomejs.dev/guides/integrate-in-editor/

(writing-tests)=

## Writing tests

This package uses [pytest][] for automated testing.
Please write {doc}`scanpy:dev/testing` for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests.
If you set up your virtual environments as described in [installing dev dependencies](#installing-dev-dependencies),
test cases should be automatically discovered by your IDE.

Alternatively, you can run all tests from the command line by executing

:::::{tab-set}
::::{tab-item} Hatch
:sync: hatch

```bash
hatch test  # test with the highest supported Python version
# or
hatch test --all  # test with all supported Python versions
```

::::

::::{tab-item} uv
:sync: uv

```bash
uv run pytest
```

::::

::::{tab-item} Pip
:sync: pip

```bash
source .venv/bin/activate
pytest
```

::::
:::::

in the root of the repository.

[pytest]: https://docs.pytest.org/

### Continuous integration

Continuous integration via GitHub actions will automatically run the tests on all pull requests and test
against the minimum and maximum supported Python version.

Additionally, there’s a CI job that tests against pre-releases of all dependencies (if there are any).
The purpose of this check is to detect incompatibilities of new package versions early on and
gives you time to fix the issue or reach out to the developers of the dependency before the package
is released to a wider audience.

The CI job is defined in `.github/workflows/test.yaml`,
however the single point of truth for CI jobs is the Hatch test matrix defined in `pyproject.toml`.
This means that local testing via hatch and remote testing on CI tests against the same python versions and uses the same environments.

### Integration testing

Cross-repo integration testing is available in the [spatialdata-integration-testing][] repo.
Please follow the instructions in its readme, which also includes a video overview.

[spatialdata-integration-testing]: https://github.com/scverse/spatialdata-integration-testing/

## Publishing a release

### Choosing the version number

`spatialdata` derives its version from the git tag through [hatch-vcs][], so cutting a release amounts to creating a tag.
Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

For pre-releases please use the `aX` suffix, such as `v0.7.0a0` or `v0.7.0a1`.
Do not use the `devX` suffix, since it does not support multiple incremental versions.
The [valid version numbers page][pypa-versioning] lists the labels you can choose from.
The naming of the tag matters: it determines whether the package is displayed as [pre-release or release](https://pypi.org/project/spatialdata/#history) on PyPI.

[hatch-vcs]: https://github.com/ofek/hatch-vcs
[pypa-versioning]: https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers

### Making a release on GitHub and publishing to PyPI

#### Recommended: create the release via GitHub

- Go to the [releases page on GitHub][releases] and press the “Draft a new release” button.
    - Press “Choose a tag” and create a new tag.
    - Please name the tag with the same string you intend for the release, including the `v` prefix.
- Alternatively, go to the [tags page on GitHub][tags], select the latest tag, and press “Create release from tag”.
    - Please name the release with the same string used for the tag (including the `v` prefix).
- Both approaches lead to the same page and view. From there:
    - Specify whether the release is a pre-release and whether it should be set as the latest release (use the checkboxes accordingly).
    - Fill in the release notes (explained in the next section).
    - Press “Publish release” to make the release available on GitHub.
- The [release workflow][] will then build the package and [upload it to PyPI](https://pypi.org/project/spatialdata/#history) using [trusted publishing][].
    - The workflow may fail; check the [release workflow status badge in the README](https://github.com/scverse/spatialdata/actions/workflows/release.yaml).

[releases]: https://github.com/scverse/spatialdata/releases
[tags]: https://github.com/scverse/spatialdata/tags
[release workflow]: https://github.com/scverse/spatialdata/blob/main/.github/workflows/release.yaml
[trusted publishing]: https://docs.pypi.org/trusted-publishers/

#### Writing release notes

We recommend using the “Generate release notes” button to automatically collect the information of all pull requests that are part of the release.
The release notes serve as a changelog for the users of the package, so it is important to have them curated and well-organized.

The automatically generated notes are grouped by change type through our [release configuration file](https://github.com/scverse/spatialdata/blob/main/.github/release.yml), which infers the type from GitHub labels and ignores pull requests opened by bots.
We recommend opening the pull requests included in the release and adding the appropriate [release labels](https://github.com/scverse/spatialdata/labels?q=release-) before generating the notes.

Use informative titles for pull requests, as these serve as the entries in the release notes; rename them if necessary.
You can also manually edit the release notes before publishing them to improve readability.

Some additional considerations:

- **Important!** If a pull request is large and its title is not informative or requires multiple lines, **do not** add a release label.
  Instead, at the end of the first message of the pull request discussion, include a markdown section titled `# Release notes` with a brief description of the intended release notes.
  This allows the person making the release to manually add the content to the release notes.
- Please avoid redundancy and do not repeat the same release notes across consecutive pre-releases, releases and post-releases.
- When generating the release notes, you can use the “Previous tag: …” button to choose which pull requests are included.
- You can see an example of a release in action in [this short video tutorial](https://www.loom.com/share/7097455bc0b9449fbe72d53fc778cbf9).

### Publishing to conda-forge

Shortly after the release lands on PyPI, a pull request is automatically opened in the conda-forge feedstock repository for the package.
The pull request contains a checklist of the tasks that need to be done before it can be merged.
Once it is merged, the new version becomes available in the conda-forge channel.

In practice, the change that is usually needed is reconciling the requirements in `pyproject.toml` with the packages and versions in the feedstock’s `meta.yaml`.
After updating `meta.yaml`, the CI runs, and if it is green the pull request can be merged.

## Writing documentation

Please write documentation for new or changed features and use-cases.
This project uses [sphinx][] with the following features:

- The [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [sphinx-autodoc-typehints][], to automatically reference annotated input and output types
- Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See scanpy’s {doc}`scanpy:dev/documentation` for more information on how to write your own.

[sphinx]: https://www.sphinx-doc.org/
[myst]: https://myst-parser.readthedocs.io/page/intro.html
[myst-nb]: https://myst-nb.readthedocs.io/
[numpydoc-napoleon]: https://www.sphinx-doc.org/page/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/page/format.html
[sphinx-autodoc-typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

(docs-building)=

### Building the docs locally

:::::{tab-set}
::::{tab-item} Hatch
:sync: hatch

```bash
hatch run docs:build
hatch run docs:open
```

::::

::::{tab-item} uv
:sync: uv

```bash
cd docs
uv run sphinx-build -M html . _build -W
(xdg-)open _build/html/index.html
```

::::

::::{tab-item} Pip
:sync: pip

```bash
source .venv/bin/activate
cd docs
sphinx-build -M html . _build -W
(xdg-)open _build/html/index.html
```

::::
:::::

## Debugging and profiling

Various tools are available to help you understand the existing code base and your own contributions.
For debugging, see the resources from [Scientific Python](https://lectures.scientific-python.org/advanced/debugging/index.html), [VS Code](https://code.visualstudio.com/docs/python/debugging) and [PyCharm](https://www.jetbrains.com/help/pycharm/debugging-your-first-python-application.html).

To find out the time or memory performance of your code, profilers can help.
Again, there are resources from [Scientific Python](https://lectures.scientific-python.org/advanced/optimizing/index.html), [napari](https://napari.org/stable/developers/contributing/performance/index.html), [PyCharm](https://www.jetbrains.com/help/pycharm/profiler.html) and [Dask](https://distributed.dask.org/en/latest/diagnosing-performance.html).
The `benchmark` dependency group and the `profiling` pixi environment declared in `pyproject.toml` provide [asv][], [memray][] and [py-spy][]; see `benchmarks/README.md` for how to run the benchmark suite.

[asv]: https://asv.readthedocs.io/
[memray]: https://bloomberg.github.io/memray/
[py-spy]: https://github.com/benfred/py-spy
