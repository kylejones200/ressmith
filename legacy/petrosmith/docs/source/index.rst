Welcome to the PetroSmith Learning Platform
=========================================

**A Comprehensive Course in Applied Petroleum Engineering**

PetroSmith is not just a Python library – it's a complete learning platform for modern petroleum engineers. This course combines classical reservoir engineering theory with practical Python implementation, preparing you to build powerful analytical tools for field operations.

.. image:: _static/oilfield_banner.png
   :alt: Petroleum Engineering
   :align: center
   :class: only-light

What You'll Learn
-----------------

This course transforms petroleum engineering education by:

✓ **Teaching fundamentals** through interactive code examples  
✓ **Building real-world skills** with production-ready calculations  
✓ **Bridging theory and practice** with validated industry methods  
✓ **Enabling rapid prototyping** of engineering workflows

.. note::
   **Prerequisites:** Basic Python knowledge and petroleum engineering fundamentals. 
   Engineers with spreadsheet experience but limited coding will find this approachable.

Course Structure
----------------

The course is organized as a progressive curriculum, taking you from foundational concepts 
through advanced field applications.

**📚 Part I: Foundations (Weeks 1-2)**

.. toctree::
   :maxdepth: 2
   :caption: Foundations

   course/module1_fundamentals
   course/module2_data_models
   course/module3_validation

**🔬 Part II: Core Disciplines (Weeks 3-6)**

.. toctree::
   :maxdepth: 2
   :caption: Core Disciplines

   course/module4_reservoir_engineering
   course/module5_drilling_engineering
   course/module6_production_engineering
   course/module7_well_completions

**🚀 Part III: Advanced Applications (Weeks 7-8)**

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   course/module8_integrated_workflows
   course/module9_optimization
   course/module10_field_development

**💼 Part IV: Professional Practice**

.. toctree::
   :maxdepth: 2
   :caption: Professional Practice

   course/exercises
   course/case_studies
   course/best_practices

**📖 Reference Materials**

.. toctree::
   :maxdepth: 2
   :caption: Reference

   installation
   api/index
   glossary
   references

Learning Outcomes
-----------------

By completing this course, you will be able to:

1. **Design Data Models** for wells, reservoirs, and production systems
2. **Implement Calculations** for reserves, flow rates, and well performance
3. **Build Engineering Tools** that replace spreadsheets with validated code
4. **Integrate Workflows** across drilling, completion, and production
5. **Optimize Operations** using systematic analytical methods

Who This Course Is For
----------------------

**Reservoir Engineers**
   Learn to automate reserves calculations, material balance, and decline curve analysis

**Drilling Engineers**
   Build tools for hydraulics, well control, and casing design

**Production Engineers**
   Create analytical systems for artificial lift, well testing, and optimization

**Completions Engineers**
   Model perforation design, tubing performance, and stimulation analysis

**Engineering Managers**
   Understand how modern tools can accelerate team productivity

Course Philosophy
-----------------

Traditional petroleum engineering education separates theory from implementation. 
You learn equations in class, then struggle to apply them correctly in Excel. 
Formulas get copied with errors, assumptions are forgotten, and debugging is painful.

**PetroSmith takes a different approach:**

.. code-block:: python

   # Traditional approach: Error-prone Excel
   # =0.0078*B2*C2*D2*E2*(1-F2)/G2  <- What are these cells?
   
   # PetroSmith approach: Self-documenting code
   from petrosmith import ReservoirAPI
   
   api = ReservoirAPI()
   reservoir = api.create_reservoir(
       reservoir_name="Main Pay",
       area=640,              # acres
       net_pay=80,            # feet
       porosity=0.22,         # fraction
       water_saturation=0.25, # fraction
       oil_fvf=1.25          # rb/stb
   )
   
   ooip = api.calculate_ooip(reservoir.reservoir_id)
   # Result: 34,135,200 STB - Reproducible, auditable, testable

**Benefits of this approach:**

- **Self-Documenting:** Variable names explain what they represent
- **Validated:** Pydantic ensures porosity can't be 1.5 or -0.2
- **Testable:** Run the same calculation 1000 times, get the same answer
- **Auditable:** SEC can review your methodology
- **Collaborative:** Share code with your team, not fragile spreadsheets

Getting Started
---------------

**Option 1: Follow the Full Course (Recommended)**

Start with :doc:`course/module1_fundamentals` and progress through each module sequentially.

**Option 2: Jump to Your Discipline**

- Reservoir engineers → :doc:`course/module4_reservoir_engineering`
- Drilling engineers → :doc:`course/module5_drilling_engineering`  
- Production engineers → :doc:`course/module6_production_engineering`

**Option 3: Quick Start**

See :doc:`installation` and :doc:`quickstart` for a 30-minute introduction.

Time Commitment
---------------

- **Full Course:** 40-60 hours (8 weeks at 5-7 hours/week)
- **Single Module:** 4-6 hours
- **Quick Start:** 30 minutes
- **Reference Use:** As needed

Assessment & Certification
--------------------------

Each module includes:

- ✓ **Learning Objectives** at the start
- ✓ **Worked Examples** throughout the chapter
- ✓ **Practice Problems** with solutions
- ✓ **Module Quiz** to test understanding
- ✓ **Capstone Project** integrating all skills

.. note::
   This is a self-paced course. Work through modules at your own speed. 
   All exercises include solutions for self-assessment.

Support & Community
-------------------

- **Documentation Issues:** `GitHub Issues <https://github.com/smithforge/petrosmith/issues>`_
- **Questions:** Engineering discussions in GitHub Discussions
- **Contributions:** Pull requests welcome for new examples or improvements

Course Credits
--------------

This course draws on industry-standard methods and validated correlations from:

- MIT OpenCourseWare - Petroleum Engineering materials
- SPE (Society of Petroleum Engineers) technical papers
- Industry textbooks by Craft & Hawkins, Ahmed, and Economides
- Field-tested methods from major operating companies

All calculations are validated against published benchmarks and commercial software.

Ready to Begin?
---------------

Start your journey into modern petroleum engineering:

👉 **Next:** :doc:`course/module1_fundamentals`

Or install PetroSmith and follow along: :doc:`installation`

---

*"The best way to learn petroleum engineering is to build something with it."*

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Course Modules

   course/module1_fundamentals
   course/module2_data_models
   course/module3_validation
   course/module4_reservoir_engineering
   course/module5_drilling_engineering
   course/module6_production_engineering
   course/module7_well_completions
   course/module8_integrated_workflows
   course/module9_optimization
   course/module10_field_development
