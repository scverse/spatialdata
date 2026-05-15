---
name: Bug report
about: Create a report to help us improve
title: ""
labels: ""
assignees: ""
---

**Recommendation: attach a minimal working example**
Generally, the easier it is for us to reproduce the issue, the faster we can work on it. It is not required, but if you can, please:

1. Reproduce using the [`blobs` dataset](https://spatialdata.scverse.org/en/stable/api/datasets.html#spatialdata.datasets.blobs)

    ```python
    from spatialdata.datasets import blobs

    sdata = blobs()
    ```

    You can also use [`blobs_annotating_element`](https://spatialdata.scverse.org/en/stable/api/datasets.html#spatialdata.datasets.blobs_annotating_element) for more
    control:

    ```
    from spatialdata.datasets import blobs_annotating_element
    sdata = blobs_annotating_element('blobs_labels')
    ```

2. If the above is not possible, reproduce using a public dataset and explain how we can download the data.
3. If the data is private, consider sharing an anonymized version/subset via a [Zulip private message](https://scverse.zulipchat.com/#user/480560), or provide screenshots/GIFs showing the behavior.

**Describe the bug**
A clear and concise description of what the bug is; please report only one bug per issue.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (optional):**

- OS: [e.g. macOS, Windows, Linux]
- Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
