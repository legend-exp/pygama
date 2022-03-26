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

Collaborators with push access that wish to release a new pygama version must
do so by following these steps:

* `Semantic versioning <https://semver.org>`_ is adopted. The version string
  uses the ``MAJOR.MINOR.PATCH`` format.
* The version string is manually specified in ``pygama/version.py``. If needed
  elsewhere in the source code (e.g. in ``setup.py``), must be read in from here.
* To release a new minor or major version, the following procedure should be
  followed:

  * The *pygama* version is updated in ``pygama/version.py``
  * A new branch with name ``releases/vMAJOR.MINOR`` (note the ``v``) containing
    the code at the intended stage is created
  * The commit is tagged with a descriptive message: ``git tag vMAJOR.MINOR.0
    -m 'short descriptive message here'`` (note the ``v``)
  * Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

* To release a new patch version, the following procedure should be followed:

  * The *pygama* version is updated in ``pygama/version.py``
  * A commit with the patch is created on the relevant release branch
    ``releases/vMAJOR.MINOR``
  * The commit is tagged: ``git tag vMAJOR.MINOR.PATCH`` (note the ``v``)
  * Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

Documentation
-------------

We adopt best practices in writing and maintaining pygama's documentation. When
contributing to the project, make sure to implement the following:

* Documentation should be exclusively available on the package website
  https://legend-exp.github.io/pygama. No READMEs, GitHub/LEGEND wiki pages
  should be written.
* Pull request authors are required to provide sufficient documentation (mainly
  as *docstrings*) for every proposed change or addition.
* Documentation for functions, classes, modules and packages should be provided
  as `Docstrings <https://peps.python.org/pep-0257>`_ along with the respective
  source code. Docstrings are automatically converted to HTML as part of the
  :ref:`pygama API documentation <pygama package>`.
* General guides, comprehensive tutorials or other high-level documentation
  (e.g. referring to how separate parts of the code interact between each
  other) must be provided as separate pages in ``docs/source/`` and linked in
  the table of contents.
* Jupyter notebooks should be added to the pygama Git repository below
  ``tutorials/``.
* Before submitting a pull request, contributors are required to build the
  documentation locally and resolve and warnings or errors.

Writing documentation
^^^^^^^^^^^^^^^^^^^^^

*In progress...*

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Scripts and tools to build documentation are located below ``docs/``. To build
documentation, the ``sphinx``, ``sphinx-rtd-theme`` and ``sphinx-multiversion``
Python packages are required. At the moment, pygama is using a ``sphinx-multiversion``
functionality only available through a fork of the project
(``samtygier-stfc/sphinx-multiversion``, at the ``prebuild_command`` branch). With pip,
it can be installed with the following command: ::

    > pip install git+https://github.com/samtygier-stfc/sphinx-multiversion.git@prebuild_command

This will probably change in the future. To build documentation for the current
Git ref, run the following commands: :: 

    > cd docs
    > make clean
    > make

Documentation can be then displayed by opening ``build/html/index.html`` with a
web browser.  To build documentation for all main pygama versions (development
branch and stable releases), run ::

    > git fetch --prune origin
    > cd docs
    > make clean
    > make allver

and display the documentation by opening ``build/allver/html/index.html``. This
documentation is also deployed to the pygama website.
