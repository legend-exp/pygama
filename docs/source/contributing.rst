Contribution guidelines
=======================

The following rules and conventions have been established for the package
development and are enforced throughout the entire code base. Merge requests
that do not comply to the following directives will be rejected.

Coding style
------------

*In progress...*

Testing
-------

* The *pygama* test suite is available below ``test/``
* Open source best practices are adopted. In particular:

  * Tests are provided for each class/method separately (*micro-testing*) e.g.
    (for each single DSP processor)
  * High-level tests of realistic applications are provided (e.g. full
    production of a real data sample)

* Unit tests are automatically run for every push event and pull request to
  the remote Git repository on a build server (currently GitHub actions)
* Every pull request must pass all tests before being approved for merging
* Pull request authors are required to provide tests with sufficient coverage
  for every proposed change or addition. If necessary, global tests (e.g. tests
  of full production chain) should be updated.

*In progress...*

Versioning
----------

Collaborators with push access to the GitHub repository that wish to release a
new project version must implement the following procedures:

* `Semantic versioning <https://semver.org>`_ is adopted. The version string
  uses the ``MAJOR.MINOR.PATCH`` format.
* The version string is manually specified in ``pygama/version.py``. If needed
  elsewhere in the source code (e.g. in ``setup.py``), must be read in from here.
* To release a new **minor** or **major version**, the following procedure
  should be followed:

  1. The *pygama* version is updated in ``pygama/version.py``
  2. A new branch with name ``releases/vMAJOR.MINOR`` (note the ``v``) containing
     the code at the intended stage is created
  3. The commit is tagged with a descriptive message: ``git tag vMAJOR.MINOR.0
     -m 'short descriptive message here'`` (note the ``v``)
  4. Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

* To release a new **patch version**, the following procedure should be followed:

  1. The *pygama* version is updated in ``pygama/version.py``
  2. A commit with the patch is created on the relevant release branch
     ``releases/vMAJOR.MINOR``
  3. The commit is tagged: ``git tag vMAJOR.MINOR.PATCH`` (note the ``v``)
  4. Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

Documentation
-------------

We adopt best practices in writing and maintaining pygama's documentation. When
contributing to the project, make sure to implement the following:

* Documentation should be exclusively available on the Project website
  https://legend-exp.github.io/pygama. No READMEs, GitHub/LEGEND wiki pages
  should be written.
* Pull request authors are required to provide sufficient documentation for
  every proposed change or addition.
* Documentation for functions, classes, modules and packages should be provided
  as `Docstrings <https://peps.python.org/pep-0257>`_ along with the respective
  source code. Docstrings are automatically converted to HTML as part of the
  :ref:`pygama API documentation <pygama package>`.
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
documentation, the ``sphinx``, ``sphinx-rtd-theme`` and ``sphinx-multiversion``
Python packages are required. At the moment, pygama is using a ``sphinx-multiversion``
functionality only available through a fork of the project
(``samtygier-stfc/sphinx-multiversion``, at the ``prebuild_command`` branch). With pip,
it can be installed with the following command:

.. code-block:: console

    $ pip install git+https://github.com/samtygier-stfc/sphinx-multiversion.git@prebuild_command

This will probably change in the future. To build documentation for the current
Git ref, run the following commands:

.. code-block:: console

    $ cd docs
    $ make clean
    $ make

Documentation can be then displayed by opening ``build/html/index.html`` with a
web browser.  To build documentation for all main pygama versions (development
branch and stable releases), run

.. code-block:: console

    $ git fetch --prune origin
    $ cd docs
    $ make clean
    $ make allver

and display the documentation by opening ``build/allver/html/index.html``. This
documentation is also deployed to the pygama website.
