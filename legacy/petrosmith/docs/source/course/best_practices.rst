Best Practices
===============

**Professional Standards for Petroleum Engineering with PetroSmith**

---

Code Quality
------------

**Always:**

✓ Use type hints for all functions  
✓ Write docstrings (Google style)  
✓ Validate inputs with Pydantic  
✓ Handle errors gracefully  
✓ Log important operations  
✓ Test calculations against known benchmarks  

**Never:**

✗ Use magic numbers without documentation  
✗ Skip input validation  
✗ Ignore units in variable names  
✗ Hard-code assumptions  
✗ Skip error handling  

---

Engineering Standards
---------------------

**Reserves Estimation:**

- Always document assumptions
- Use P10/P50/P90 for uncertainty
- Follow SPE PRMS guidelines
- Maintain audit trail
- Benchmark against analogs

**Well Design:**

- Apply appropriate safety factors
- Consider worst-case scenarios
- Verify against industry standards (API, ISO)
- Document design basis
- Peer review critical wells

**Production Forecasting:**

- Validate decline curves against history
- Use multiple methods (DCA, type curves, analogs)
- Consider interference effects
- Account for operational constraints
- Update forecasts regularly

---

Documentation
-------------

**Every Analysis Should Include:**

1. **Objective:** What are you trying to determine?
2. **Data Sources:** Where did the data come from?
3. **Assumptions:** What did you assume and why?
4. **Methodology:** What calculations did you perform?
5. **Results:** What did you find?
6. **Recommendations:** What should be done?
7. **Sensitivity:** How sensitive are results to key inputs?
8. **Peer Review:** Who reviewed this?

---

Version Control
---------------

**Use Git for:**

- All analysis code
- Configuration files
- Documentation
- Jupyter notebooks

**Commit messages should:**

- Describe what changed and why
- Reference tickets/issues
- Tag major milestones

---

Testing
-------

**Types of Tests:**

1. **Unit Tests:** Individual calculations
2. **Integration Tests:** Workflows
3. **Validation Tests:** Against benchmarks
4. **Regression Tests:** Ensure changes don't break existing functionality

**Example:**

.. code-block:: python

   import pytest
   from petrosmith import ReservoirAPI
   
   def test_ooip_calculation():
       """Test OOIP against hand calculation."""
       api = ReservoirAPI()
       
       # Known result from textbook example
       ooip = api.calculate_ooip(
           reservoir_id="TEST",
           area=640,
           oil_fvf=1.25
       )
       
       expected = 49_297_000  # STB
       assert abs(ooip - expected) / expected < 0.01  # Within 1%

---

Security
--------

**Protect Sensitive Data:**

- Never commit API keys or passwords
- Use environment variables for credentials
- Encrypt production data
- Follow corporate data policies

---

Performance
-----------

**Optimize for:**

- Large datasets (100K+ wells)
- Real-time calculations
- Repeated analysis

**Techniques:**

- Vectorize with NumPy
- Cache expensive calculations
- Use appropriate data structures
- Profile before optimizing

---

Collaboration
-------------

**Working in Teams:**

- Use consistent coding style (Black, PEP 8)
- Review each other's code
- Share reusable functions
- Document tribal knowledge
- Conduct knowledge transfer sessions

---

**Next:** :doc:`../references`
