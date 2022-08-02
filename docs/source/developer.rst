Developer's guide
=================

The following rules and conventions have been established for the package
development and are enforced throughout the entire code base. Merge requests
that do not comply to the following directives will be rejected.

To start developing :mod:`pygama`, clone the remote repository:

.. code-block:: console

  $ git clone https://github.com/legend-exp/pygama

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

Documentation
-------------

We adopt best practices in writing and maintaining :mod:`pygama`'s
documentation. When contributing to the project, make sure to implement the
following:

* Documentation should be exclusively available on the Project website
  https://legend-exp.github.io/pygama. No READMEs, GitHub/LEGEND wiki pages
  should be written.
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

Building documentation
^^^^^^^^^^^^^^^^^^^^^^

Scripts and tools to build documentation are located below ``docs/``. To build
documentation, ``sphinx`` and a couple of additional Python packages are
required. You can get all the needed dependencies by running:

.. code-block:: console

  $ cd pygama
  $ pip install '.[docs]'

To build documentation for the current Git ref, run the following commands:

.. code-block:: console

  $ cd docs
  $ make clean
  $ make

Documentation can be then displayed by opening ``build/html/index.html`` with a
web browser.  To build documentation for all main :mod:`pygama` versions
(development branch and stable releases), run

.. code-block:: console

  $ git fetch --prune origin
  $ cd docs
  $ make clean
  $ make allver

and display the documentation by opening ``build/allver/html/index.html``. This
documentation is also deployed to the :mod:`pygama` website.

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
