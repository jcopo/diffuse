Installation
============

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install diffuse

Development Installation
------------------------

For development:

.. code-block:: bash

   git clone https://github.com/jcopo/diffuse.git
   cd diffuse
   pip install -e .

JAX Installation
----------------

For GPU support:

.. code-block:: bash

   pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Verification
------------

Test your installation:

.. code-block:: python

   import jax
   import diffuse
   print(f"JAX backend: {jax.default_backend()}")

Next Steps
----------

See the :doc:`quickstart` guide to get started.