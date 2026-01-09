ResSmith follows a four layer architecture.

The first layer holds objects. Objects represent production series, forecast specifications, decline parameters, and economic assumptions. Objects never implement algorithms.

The second layer holds primitives. Primitives implement decline models, fitting routines, preprocessing steps, and economic calculations. Primitives never load files and never plot results.

The third layer holds tasks. Tasks bind intent. A task defines what it means to fit a decline, forecast production, or evaluate economics. Tasks validate inputs and orchestrate primitives.

The fourth layer holds workflows. Workflows handle reality. They accept pandas inputs, manage batch runs, save outputs, and integrate with plotting and reporting tools.

Each layer imports only from layers below it. This structure prevents decline logic from mixing with economics or I O concerns.

ResSmith reuses temporal typing from Timesmith. This allows decline workflows to integrate cleanly with anomaly detection, visualization, and asset health systems.

