# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this repo, too.
This document will not reproduce the entire content from there. Instead, it aims at summarizing the most important
information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub. If not, please refer
to the [scanpy developer guide][].

## Installing dev dependencies

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build
the documentation_. It's easy to install them using `pip`:

```bash
cd spatialdata-io
pip install -e ".[dev,test,doc]"
```

## Code-style

This template uses [pre-commit][] to enforce consistent code-styles. On every commit, pre-commit checks will either
automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository. Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub. If you didn't run `pre-commit` before
pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

Finally, most editors have an _autoformat on save_ feature. Consider enabling this option for [black][black-editors]
and [prettier][prettier-editors].

[black-editors]: https://black.readthedocs.io/en/stable/integrations/editors.html
[prettier-editors]: https://prettier.io/docs/en/editors.html

## Writing tests

```{note}
Remember to first install the package with `pip install '-e[dev,test]'`
```

This package uses [pytest][] for automated testing. Please [write tests][scanpy-test-docs] for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively, you can run all tests from the command line by executing

```bash
pytest
```

in the root of the repository. Continuous integration will automatically run the tests on all pull requests.

### Continuous integration

Continuous integration will automatically run the tests on all pull requests and test against the minimum and maximum supported Python version.

Additionally, there's a CI job that tests against pre-releases of all dependencies (if there are any). The purpose of this check is to detect incompatibilities of new package versions early on and gives you time to fix the issue or reach out to the developers of the dependency before the package is released to a wider audience.

[scanpy-test-docs]: https://scanpy.readthedocs.io/en/latest/dev/testing.html#writing-tests

By including this additional information, the document now provides a more comprehensive overview of the continuous integration process related to testing.

### Integration testing

Cross-repo integration testing is available in the [spatialdata-integration-testing](https://github.com/scverse/spatialdata-integration-testing/) repo. Please follow the instructions in the Readme (which also includes a video overview).

## Publishing a release

### Updating the version number

Before making a release, you need to update the version number. Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1.  MAJOR version when you make incompatible API changes,
> 2.  MINOR version when you add functionality in a backwards compatible manner, and
> 3.  PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.
> For pre-release please use the aX suffix, such as v0.7.0a0, v0.7.0a1. Do not use the devX suffix since it doesn't support multiple incremental versions.

You can find the [labels for pre-release in this page](https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers).

You can either use [bump2version][] to automatically create a git tag with the updated version number, or manually create the tag yourself (locally or from the GitHub interface when making a release).
If you use `bump2version`, you can run one of the following commands in the root of the repository

```bash
bump2version patch
bump2version minor
bump2version major
```

Once you are done, run

```
git push --tags
```

to publish the created tag on GitHub.

It's important that the tag for a pre-release follows this naming convention as it will determine if the package is displayed as [pre-release or release](https://pypi.org/project/spatialdata/#history) in PyPI.

[bump2version]: https://github.com/c4urself/bump2version

### Making a release on GitHub and publishing to PyPI

#### Recommended: Create the release via GitHub

- Go to the [Releases page on GitHub](https://github.com/scverse/spatialdata/releases) and press the “Draft a new release” button.
    - Press “Choose a tag” and create a new tag.
    - Please name the tag with the same string you intend for the release, including the `v` prefix.
- Alternatively, go to the [Tags page on GitHub](https://github.com/scverse/spatialdata/tags), select the latest tag, and press “Create release from tag”.
    - Please name the release with the same string used for the tag (including the `v` prefix).
- Both approaches lead to the same page and view. From there:
    - Specify whether the release is a pre-release and whether it should be set as the latest release (use the checkboxes accordingly).
    - Fill in the release notes (explained in the next section).
    - Press “Publish release” to make the release available on GitHub.
- A [GitHub Action](https://github.com/scverse/spatialdata/blob/main/.github/workflows/release.yaml) will automatically build the package and [upload it to PyPI](https://pypi.org/project/spatialdata/#history).
    - The action may fail; check the [workflow status badge in the README](https://github.com/scverse/spatialdata/actions/workflows/release.yaml).

#### Not recommended: Manual tag-first workflow

- If you already tagged and pushed a commit as explained above and want to create a release from that tag, you can go to the [Tags page on GitHub](https://github.com/scverse/spatialdata/tags), select the latest tag, and press “Create release from tag”.
    - Please name the release with the same string used for the tag (including the `v` prefix).

#### Writing release notes

We recommend using the button "Generate release notes" to automatically collect all the information of the pull requests that are part of the release.
The release notes serve as a changelog for the user of the package so it's important to have them curated and well-organized. This is explained in depth below.

Here is an example of automatically generated release notes for a previous release (v0.2.3):

```
## What's Changed
* Add clip parameter to polygon_query; tests missing by @LucaMarconato in https://github.com/scverse/spatialdata/pull/670
* Add sort parameter to points model by @LucaMarconato in https://github.com/scverse/spatialdata/pull/672
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/scverse/spatialdata/pull/673
* Docs for datasets (blobs, raccoon) by @LucaMarconato in https://github.com/scverse/spatialdata/pull/674
* Update issue templates by @LucaMarconato in https://github.com/scverse/spatialdata/pull/675
* Minor fixes: `id()` -> `is`, inplace category subset `AnnData` relational query by @LucaMarconato in https://github.com/scverse/spatialdata/pull/681
* Added ColorLike to _types.py by @timtreis in https://github.com/scverse/spatialdata/pull/689
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/scverse/spatialdata/pull/685
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/scverse/spatialdata/pull/690
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/scverse/spatialdata/pull/698
* Fix labels multiscales method by @aeisenbarth in https://github.com/scverse/spatialdata/pull/697


**Full Changelog**: https://github.com/scverse/spatialdata/compare/v0.2.2...v0.2.3
```

The release notes above can be hard to read, but this is addressed by our [configuration file](https://github.com/scverse/spatialdata/blob/main/.github/release.yml). It organizes release notes by change type, inferred from GitHub labels, and ignores PRs from bots. We recommend opening the PRs included in the release and adding the appropriate labels. The automatic generation will then group PRs by [release labels](https://github.com/scverse/spatialdata/labels?q=release-) and list each PR on a separate line. Here is an example output:

```
<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Major
* Adding `attrs` at the `SpatialData` object level by @quentinblampey in https://github.com/scverse/spatialdata/pull/711
### Minor
* Add asv benchmark code by @berombau in https://github.com/scverse/spatialdata/pull/784
* relabel block by @ArneDefauw in https://github.com/scverse/spatialdata/pull/664
* validate tables while parsing by @melonora in https://github.com/scverse/spatialdata/pull/808
### Fixed
* relaxed fsspec version by @LucaMarconato in https://github.com/scverse/spatialdata/pull/798
* fix for to_polygons when using processes instead of threads in dask by @ArneDefauw in https://github.com/scverse/spatialdata/pull/756
* Fix `transform_to_data_extent` converting labels to images by @aeisenbarth in https://github.com/scverse/spatialdata/pull/791
* fix join non matching table by @melonora in https://github.com/scverse/spatialdata/pull/813


**Full Changelog**: https://github.com/scverse/spatialdata/compare/v0.2.6...v0.2.7
```

Use informative titles for PRs, as these will serve as section titles in the release notes (rename the PRs if necessary). You can also manually edit the release notes before publishing them to improve readability.

Some additional considerations

- **Important!** If a PR is large and its title isn't informative or requires multiple lines, **do not** add a release tag. Instead, at the end of the first message of the PR discussion, please include a markdown section with title `# Release notes` with a brief description of the intended release notes. This will allow the person making a release to manually add the PR content to the release notes during the release process.
- Please avoid redundancy and do not add the same release notes to consecutive pre-releases/releases/post-releases.
- When automatically generating the release notes, you can use the button "Previous tag: ..." to choose which PRs will be included in the release notes.
- Finally, you can see an example of a release in action in from Luca [this short video tutorial](https://www.loom.com/share/7097455bc0b9449fbe72d53fc778cbf9).

### Publishing to conda-forge

Shortly after you make a release in PyPI, a new PR will be automatically made in the conda-forge "feedstock repository" for the package (this has been previously setup). The PR will contain a checklist of which tasks should be done to be able to merge the PR. Once the PR is merged, the package will be available in the conda-forge channel.

Practically, the changes that usually needs to be done are comparing the package requirements in `pyproject.toml` from your repository, with the packages and versions in the `meta.yaml` file in the conda-forge feedstock repository. If there are any differences, you should update the `meta.yaml` file accordingly. After that, the CI will run and if green the PR can be merged.

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

- the [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [Sphinx autodoc typehints][], to automatically reference annotated input and output types

See the [scanpy developer docs](https://scanpy.readthedocs.io/en/latest/dev/documentation.html) for more information
on how to write documentation.

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your reponsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration, please check
out [this feature request](https://github.com/scverse/cookiecutter-scverse/issues/40) in the `cookiecutter-scverse`
repository.

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`. Only
  if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control, you can add an entry to
  the `nitpick_ignore` list in `docs/conf.py`

#### Building the docs locally

```bash
cd docs
make html
open _build/html/index.html
```

### Debugging and profiling

There are various tools available to help you understand the existing code base and your new code contributions. For debugging code there are multiple resources available: [Scientific Python](https://lectures.scientific-python.org/advanced/debugging/index.html), [VSCode](https://code.visualstudio.com/docs/python/debugging) and [PyCharm](https://www.jetbrains.com/help/pycharm/debugging-your-first-python-application.html).

To find out the time or memory performance of your code, profilers can help. Again, various resources from [Scientific Python](https://lectures.scientific-python.org/advanced/optimizing/index.html), [napari](https://napari.org/stable/developers/contributing/performance/index.html), [PyCharm](https://www.jetbrains.com/help/pycharm/profiler.html) and [Dask](https://distributed.dask.org/en/latest/diagnosing-performance.html) can be helpful.

<!-- Links -->

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html
[github quickstart guide]: https://docs.github.com/en/get-started/quickstart/create-a-repo?tool=webui
[codecov]: https://about.codecov.io/sign-up/
[codecov docs]: https://docs.codecov.com/docs
[codecov bot]: https://docs.codecov.com/docs/team-bot
[codecov app]: https://github.com/apps/codecov
[pre-commit.ci]: https://pre-commit.ci/
[readthedocs.org]: https://readthedocs.org/
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[pre-commit]: https://pre-commit.com/
[anndata]: https://github.com/scverse/anndata
[mudata]: https://github.com/scverse/mudata
[pytest]: https://docs.pytest.org/
[semver]: https://semver.org/
[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
[pypi]: https://pypi.org/
