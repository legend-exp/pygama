Installation
============

Install on local systems with:

.. code-block:: console

    $ git clone https://github.com/legend-exp/pygama
    $ pip install -e pygama

You can omit the ``-e`` flag (editable mode) if you do not plan to develop
pygama. If you do not have admin rights you can install pygama as a normal
user:

.. code-block:: console

    $ pip install -e pygama --user

and make sure that your ``PYTHONPATH`` environment variable is aware of the
install location. To uninstall pygama run:

.. code-block:: console

    $ pip uninstall pygama
