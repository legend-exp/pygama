Developer's guide
=================

The following rules and conventions have been established for the package
development and are enforced throughout the entire code base. Merge requests
that do not comply to the following directives will be rejected.

To start developing :mod:`pygama`, fork the remote repository to your personal
GitHub account (see `About Forks <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks>`_).
If you have not set up your ssh keys on the computer you will be working on,
please follow `GitHub's instructions <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_. Once you have your own fork, you can clone it via
(replace "yourusername" with your GitHub username):

.. code-block:: console

  $ git clone git@github.com:yourusername/pygama.git

All extra tools needed to develop :mod:`pygama` are listed as optional
dependencies and can be installed via pip by running:

.. code-block:: console

  $ cd pygama
  $ pip install '.[all]'  # single quotes are not needed on bash

Code style
----------

* All functions and methods (arguments and return types) must be
  `type-annotated <https://docs.python.org/3/library/typing.html>`_. Type
  annotations for variables like class attributes are also highly appreciated.
  Do not forget to

  .. code-block:: python

    from __future__ import annotations

  at the top of a module implementation.
* Messaging to the user is managed through the :mod:`logging` module. Do not
  add :func:`print` statements. To make a logging object available in a module,
  add this:

  .. code-block:: python

    import logging
    log = logging.getLogger(__name__)

  at the top. In general, try to keep the number of :func:`logging.debug` calls
  low and use informative messages. :func:`logging.info` calls should be
  reserved for messages from high-level routines (like
  :func:`pygama.dsp.build_dsp`). Good code is never too verbose.
* If an error condition leading to undefined behavior occurs, raise an
  exception. try to find the most suitable between the `built-in exceptions
  <https://docs.python.org/3/library/exceptions.html>`_, otherwise ``raise
  RuntimeError("message")``. Do not raise ``Warning``\ s, use
  :func:`logging.warning` for that and don't abort the execution.
* Warning messages (emitted when a problem is encountered that does not lead to
  undefined behavior) must be emitted through :func:`logging.warning` calls.

A set of `pre-commit <https://pre-commit.com>`_ hooks is configured to make
sure that :mod:`pygama` coherently follows standard coding style conventions.
The pre-commit tool is able to identify common style problems and automatically
fix them, wherever possible. Configured hooks are listed in the
``.pre-commit-config.yaml`` file at the project root folder. They are run
remotely on the GitHub repository through the `pre-commit bot
<https://pre-commit.ci>`_, but can also be run locally before submitting a
pull request (recommended):

.. code-block:: console

  $ cd pygama
  $ pip install '.[test]'
  $ pre-commit run --all-files  # analyse the source code and fix it wherever possible
  $ pre-commit install          # install a Git pre-commit hook (optional but recommended)

For a more comprehensive guide, check out the `Scikit-HEP documentation about
code style <https://scikit-hep.org/developer/style>`_.

Testing
-------

* The :mod:`pygama` test suite is available below ``tests/``. We use `pytest
  <https://docs.pytest.org>`_ to run tests and analyze their output. As
  a starting point to learn how to write good tests, reading of `the
  Scikit-HEP Intro to testing <https://scikit-hep.org/developer/pytest>`_ is
  recommended. Refer to `pytest's how-to guides
  <https://docs.pytest.org/en/stable/how-to/index.html>`_ for a complete
  overview.
* :mod:`pygama` tests belong to three categories:

  :unit tests: Should ensure the correct behaviour of each function
      independently, possibly without relying on other :mod:`pygama` methods.
      The existence of these micro-tests makes it possible to promptly identify
      and fix the source of a bug. An example of this are tests for each single
      DSP processor

  :integration tests: Should ensure that independent parts of the code base
      work well together and are integrated in a cohesive framework. An example
      of this is testing whether :func:`moduleA.process_obj` is able to
      correctly handle :class:`moduleB.DataObj`

  :functional tests: High-level tests of realistic applications. An example is
      testing whether the processing of a real or synthetic data sample yields
      consistent output parameters

* Unit tests are automatically run for every push event and pull request to the
  remote Git repository on a remote server (currently handled by GitHub
  actions). Every pull request must pass all tests before being approved for
  merging. Running the test suite is simple:

  .. code-block:: console

    $ cd pygama
    $ pip install '.[test]'
    $ pytest

* Additionally, pull request authors are required to provide tests with
  sufficient code coverage for every proposed change or addition. If necessary,
  high-level functional tests should be updated. We currently rely on
  `codecov.io <https://app.codecov.io/gh/legend-exp/pygama>`_ to keep track of
  test coverage. A local report, which must be inspected before submitting pull
  requests, can be generated by running:

  .. code-block:: console

    $ pytest --cov=pygama

Testing Numba-Wrapped Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using Numba to vectorize Python functions, the Python version of the function
does not, by default, get directly tested, but the Numba version instead. In
this case, we need to unwrap the Numba function and test the pure Python version.
With various processors in :mod:`pygama.dsp.processors`, this means that testing
and triggering the code coverage requires this unwrapping.

Within the testing suite, we use the :func:`@pytest.fixture()<pytest.fixture>`
decorator to include a helper function called ``compare_numba_vs_python`` that
can be used in any test. This function runs both the Numba and pure Python versions
of a function, asserts that they are equal up to floating precision, and returns the
output value.

As an example, we show a snippet from the test for
:func:`pygama.dsp.processors.fixed_time_pickoff`, a processor which uses the
:func:`@numba.guvectorize()<numba.guvectorize>` decorator.

.. code-block:: python

    def test_fixed_time_pickoff(compare_numba_vs_python):
        """Testing function for the fixed_time_pickoff processor."""

        len_wf = 20

        # test for nan if w_in has a nan
        w_in = np.ones(len_wf)
        w_in[4] = np.nan
        assert np.isnan(compare_numba_vs_python(fixed_time_pickoff, w_in, 1, ord("i")))

In the assertion that the output is what we expect, we use
``compare_numba_vs_python(fixed_time_pickoff, w_in, 1, ord("i"))`` in place of
``fixed_time_pickoff(w_in, 1, ord("i"))``. In general, the replacement to make is
``func(*inputs)`` becomes ``compare_numba_vs_python(func, *inputs)``.

Note, that in cases of testing for the raising of errors, it is recommended
to instead run the function twice: once with the Numba version, and once using the
:func:`inspect.unwrap` function. We again show a snippet from the test for
:func:`pygama.dsp.processors.fixed_time_pickoff` below. We include the various
required imports in the snippet for verbosity.

.. code-block:: python

    import inspect

    import numpy as np
    import pytest

    from pygama.dsp.errors import DSPFatal
    from pygama.dsp.processors import fixed_time_pickoff

    def test_fixed_time_pickoff(compare_numba_vs_python):
    "skipping parts of function..."
    # test for DSPFatal errors being raised
    # noninteger t_in with integer interpolation
    with pytest.raises(DSPFatal):
        w_in = np.ones(len_wf)
        fixed_time_pickoff(w_in, 1.5, ord("i"))

    with pytest.raises(DSPFatal):
        a_out = np.empty(len_wf)
        inspect.unwrap(fixed_time_pickoff)(w_in, 1.5, ord("i"), a_out)

In this case, the general idea is to use :func:`pytest.raises` twice, once with
``func(*inputs)``, and again with ``inspect.unwrap(func)(*inputs)``.

Testing Factory Functions that Return Numba-Wrapped Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As in the previous section, we also have processors that are first initialized
with a factory function, which then returns a callable Numba-wrapped function.
In this case, there is a slightly different way of testing the function to ensure
full code coverage when using ``compare_numba_vs_python``, as the function
signature is generally different.

As an example, we show a snippet from the test for
:func:`pygama.dsp.processors.dwt.discrete_wavelet_transform`, a processor which uses
a factory function to return a function wrapped by the
:func:`@numba.guvectorize()<numba.guvectorize>` decorator.

.. code-block:: python

    import numpy as np
    import pytest

    from pygama.dsp.errors import DSPFatal
    from pygama.dsp.processors import discrete_wavelet_transform

    def test_discrete_wavelet_transform(compare_numba_vs_python):
        """Testing function for the fixed_time_pickoff processor."""

        # set up values to use for each test case
        len_wf_in = 16
        wave_type = 'haar'
        level = 2
        len_wf_out = 4

        # ensure the DSPFatal is raised for a negative level
        with pytest.raises(DSPFatal):
            discrete_wavelet_transform(wave_type, -1)

        # ensure that a valid input gives the expected output
        w_in = np.ones(len_wf_in)
        w_out = np.empty(len_wf_out)
        w_out_expected = np.ones(len_wf_out) * 2**(level / 2)

        dwt_func = discrete_wavelet_transform(wave_type, level)
        assert np.allclose(
            compare_numba_vs_python(dwt_func, w_in, w_out),
            w_out_expected,
        )
        ## rest of test function is truncated in this example

In this case, the error is raised outside of the Numba-wrapped function, and
we only need to test for the error once. For the comparison of the calculated
values to expectation, we must initialize the output array and pass it to the
list of inputs that should be used in the comparison. This is different than
the previous section, where we are instead now updating the outputted values
in place.

Documentation
-------------

We adopt best practices in writing and maintaining :mod:`pygama`'s
documentation. When contributing to the project, make sure to implement the
following:

* Documentation should be exclusively available on the Project website
  `pygama.readthedocs.io <https://pygama.readthedocs.io>`_. No READMEs,
  GitHub/LEGEND wiki pages should be written.
* Pull request authors are required to provide sufficient documentation for
  every proposed change or addition.
* Documentation for functions, classes, modules and packages should be provided
  as `Docstrings <https://peps.python.org/pep-0257>`_ along with the respective
  source code. Docstrings are automatically converted to HTML as part of the
  :mod:`pygama` package API documentation.
* General guides, comprehensive tutorials or other high-level documentation
  (e.g. referring to how separate parts of the code interact between each
  other) must be provided as separate pages in ``docs/source/`` and linked in
  the table of contents.
* Jupyter notebooks should be added to the main Git repository below
  ``tutorials/``.
* Before submitting a pull request, contributors are required to build the
  documentation locally and resolve and warnings or errors.

Writing documentation
^^^^^^^^^^^^^^^^^^^^^

We adopt the following guidelines for writing documentation:

* Documentation source files must formatted in reStructuredText (reST). A
  reference format specification is available on the `Sphinx reST usage guide
  <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_.
  Usage of `Cross-referencing syntax
  <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#cross-referencing-syntax>`_
  in general and `for Python objects
  <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects>`_
  in particular is recommended. We also support cross-referencing external
  documentation via `sphinx.ext.intersphinx
  <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_,
  when referring for example to :class:`pandas.DataFrame`.
* To document Python objects, we also adopt the `NumPy Docstring style
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_. Examples are
  available `here
  <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
* We support also the Markdown format through the `MyST-Parser
  <https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html>`_.
* Jupyter notebooks placed below ``docs/source/notebooks`` are automatically
  rendered to HTML pages by the `nbsphinx <https://nbsphinx.readthedocs.io>`_
  extension.

Building documentation
^^^^^^^^^^^^^^^^^^^^^^

Scripts and tools to build documentation are located below ``docs/``. To build
documentation, ``sphinx`` and a couple of additional Python packages are
required. You can get all the needed dependencies by running:

.. code-block:: console

  $ cd pygama
  $ pip install '.[docs]'

`Pandoc <https://pandoc.org/installing.html>`_ is also required to render
Jupyter notebooks. To build documentation, run the following commands:

.. code-block:: console

  $ cd docs
  $ make clean
  $ make

Documentation can be then displayed by opening ``build/html/index.html`` with a
web browser. Documentation for the :mod:`pygama` website is built and deployed by
`Read the Docs <https://readthedocs.org/projects/pygama>`_.

Versioning
----------

Collaborators with push access to the GitHub repository that wish to release a
new project version must implement the following procedures:

* `Semantic versioning <https://semver.org>`_ is adopted. The version string
  uses the ``MAJOR.MINOR.PATCH`` format.
* To release a new **minor** or **major version**, the following procedure
  should be followed:

  1. A new branch with name ``releases/vMAJOR.MINOR`` (note the ``v``) containing
     the code at the intended stage is created
  2. The commit is tagged with a descriptive message: ``git tag vMAJOR.MINOR.0
     -m 'short descriptive message here'`` (note the ``v``)
  3. Changes are pushed to the remote:

     .. code-block:: console

       $ git push origin releases/vMAJOR.MINOR
       $ git push origin refs/tags/vMAJOR.MINOR.0

* To release a new **patch version**, the following procedure should be followed:

  1. A commit with the patch is created on the relevant release branch
     ``releases/vMAJOR.MINOR``
  2. The commit is tagged: ``git tag vMAJOR.MINOR.PATCH`` (note the ``v``)
  3. Changes are pushed to the remote:

     .. code-block:: console

       $ git push origin releases/vMAJOR.MINOR
       $ git push origin refs/tags/vMAJOR.MINOR.PATCH

* To upload the release to the `Python Package Index
  <https://pypi.org/project/pygama>`_, a new release must be created through
  `the GitHub interface <https://github.com/legend-exp/pygama/releases/new>`_,
  associated to the just created tag.  Usage of the "Generate release notes"
  option is recommended.
