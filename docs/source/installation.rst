Installation Guide
==================

Requirements
-----------

- Python 3.12 or newer
- pip package manager

Quick Install
------------

.. code-block:: bash

   pip install petrosmith

From Source
----------

.. code-block:: bash

   git clone https://github.com/your-org/petrosmith.git
   cd petrosmith
   pip install -e .

Verify Installation
------------------

.. code-block:: python

   import petrosmith
   print(petrosmith.__version__)
   
   # Run quick test
   from petrosmith.api import DrillingAPI
   api = DrillingAPI()
   pressure = api.calculate_hydrostatic_pressure(10.5, 8000)
   print(f"Hydrostatic pressure: {pressure:.0f} psi")

Expected output: ``Hydrostatic pressure: 4368 psi``

Dependencies
-----------

PetroSmith automatically installs:

- numpy (numerical computations)
- scipy (scientific calculations)
- pydantic (data validation)
- python-dateutil (date handling)

IDE Setup
---------

**VS Code** (Recommended)

Install Python extension, then add to ``.vscode/settings.json``:

.. code-block:: json

   {
       "python.linting.enabled": true,
       "python.linting.pylintEnabled": true
   }

**PyCharm**

Works out of the box. Mark ``petrosmith`` as sources root.

**Jupyter Notebook**

.. code-block:: bash

   pip install jupyter
   jupyter notebook

Great for interactive analysis and visualization of results.

Troubleshooting
--------------

**Import Error**

If you see ``ModuleNotFoundError: No module named 'petrosmith'``:

.. code-block:: bash

   # Verify installation
   pip list | grep petrosmith
   
   # Reinstall if needed
   pip install --force-reinstall petrosmith

**Version Conflicts**

.. code-block:: bash

   # Create clean virtual environment
   python -m venv petro_env
   source petro_env/bin/activate  # On Windows: petro_env\\Scripts\\activate
   pip install petrosmith

Next Steps
----------

- :doc:`quickstart` - Your first calculations
- :doc:`business_value` - Why this matters
- :doc:`examples` - Real-world examples
