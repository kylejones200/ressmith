"""Report generation module for reservoir engineering analysis.

This module provides single-page summary reports for wells and fields,
including HTML and PDF output formats.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ressmith.objects.domain import ForecastResult
from ressmith.primitives.diagnostics import FitDiagnostics

logger = logging.getLogger(__name__)

# Try to import reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.debug(
        "reportlab not available. Install with: pip install reportlab. "
        "PDF generation will not be available."
    )


def generate_well_report(
    forecast: ForecastResult,
    diagnostics: FitDiagnostics | None = None,
    params: dict[str, float] | None = None,
    output_path: str | None = None,
    format: str = "html",
    well_id: str | None = None,
) -> str:
    """Generate single-page well report.

    Args:
        forecast: ForecastResult object
        diagnostics: Optional FitDiagnostics
        params: Optional model parameters dictionary
        output_path: Output file path (auto-generated if None)
        format: Output format ('html' or 'pdf')
        well_id: Well identifier for filename

    Returns:
        Path to generated report file

    Example:
        >>> from ressmith import fit_forecast
        >>> from ressmith.workflows.reports import generate_well_report
        >>> forecast, params = fit_forecast(data)
        >>> report_path = generate_well_report(forecast, params=params)
        >>> print(f"Report saved to {report_path}")
    """
    if output_path is None:
        well_id_str = well_id or "well_001"
        output_path = f"{well_id_str}_report.html"

    logger.info(f"Generating well report: {output_path} (format: {format})")

    format_handlers = {
        "pdf": lambda: _handle_pdf_report(
            forecast, diagnostics, params, output_path, well_id
        ),
        "html": lambda: _handle_html_report(
            forecast, diagnostics, params, output_path, well_id
        ),
    }

    handler = format_handlers.get(format, format_handlers["html"])
    return handler()


def _handle_pdf_report(
    forecast: ForecastResult,
    diagnostics: FitDiagnostics | None,
    params: dict[str, float] | None,
    output_path: str,
    well_id: str | None,
) -> str:
    """Handle PDF report generation."""
    if not REPORTLAB_AVAILABLE:
        logger.warning(
            "reportlab not available. Install with: pip install reportlab. "
            "Saving HTML instead."
        )
        return _handle_html_report(forecast, diagnostics, params, output_path, well_id)

    pdf_path = (
        output_path.replace(".html", ".pdf")
        if output_path.endswith(".html")
        else output_path
    )
    if not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"
    _generate_pdf_report(forecast, diagnostics, params, pdf_path, well_id)
    return pdf_path


def _handle_html_report(
    forecast: ForecastResult,
    diagnostics: FitDiagnostics | None,
    params: dict[str, float] | None,
    output_path: str,
    well_id: str | None,
) -> str:
    """Handle HTML report generation."""
    html_content = _generate_html_report(forecast, diagnostics, params, well_id)
    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def _generate_pdf_report(
    forecast: ForecastResult,
    diagnostics: FitDiagnostics | None,
    params: dict[str, float] | None,
    output_path: str,
    well_id: str | None,
    plot_path: str | None = None,
) -> None:
    """Generate PDF report using reportlab."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not available. Install with: pip install reportlab"
        )

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title = Paragraph("Reservoir Engineering Analysis Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    if well_id:
        story.append(Paragraph(f"<b>Well ID:</b> {well_id}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

    if params:
        story.append(Paragraph("<b>Model Parameters</b>", styles["Heading2"]))
        param_data = [["Parameter", "Value"]]
        param_data.extend([[param, f"{value:.4f}"] for param, value in params.items()])

        param_table = Table(param_data)
        param_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(param_table)
        story.append(Spacer(1, 0.2 * inch))

    if diagnostics:
        story.append(Paragraph("<b>Diagnostics</b>", styles["Heading2"]))
        diag_data = [
            ["Metric", "Value"],
            ["RMSE", f"{diagnostics.rmse:.4f}"],
            ["MAE", f"{diagnostics.mae:.4f}"],
            ["MAPE", f"{diagnostics.mape:.4f}"],
            ["R²", f"{diagnostics.r_squared:.4f}"],
        ]

        diag_table = Table(diag_data)
        diag_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(diag_table)
        story.append(Spacer(1, 0.2 * inch))

    if plot_path and Path(plot_path).exists():
        try:
            img = Image(plot_path, width=6 * inch, height=4 * inch)
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("<b>Forecast Plot</b>", styles["Heading2"]))
            story.append(img)
        except Exception as e:
            logger.warning(f"Could not embed plot: {e}")

    doc.build(story)
    logger.info(f"PDF report saved to {output_path}")


def generate_field_pdf_report(
    well_results: list[dict],
    output_path: str | None = None,
    title: str = "Field Summary Report",
) -> str:
    """Generate field summary PDF report."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not available. Install with: pip install reportlab"
        )

    if output_path is None:
        output_path = "field_summary.pdf"

    df = pd.DataFrame(well_results)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_para = Paragraph(title, styles["Title"])
    story.append(title_para)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Summary Statistics</b>", styles["Heading2"]))
    summary_data = [["Metric", "Value"]]
    summary_data.append(["Total Wells", str(len(df))])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sum_metrics = {"eur", "npv", "p50_eur", "p50_npv"}
    for col in numeric_cols[:10]:
        if col in sum_metrics:
            summary_data.append([f"{col} (Total)", f"{df[col].sum():,.0f}"])
            summary_data.append([f"{col} (Mean)", f"{df[col].mean():,.0f}"])

    summary_table = Table(summary_data)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Well Results (Sample)</b>", styles["Heading2"]))
    well_data = [list(df.columns[:8])]
    well_data.extend(
        [
            [str(row[col])[:20] for col in df.columns[:8]]
            for _, row in df.head(20).iterrows()
        ]
    )

    well_table = Table(well_data)
    well_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(well_table)

    if len(df) > 20:
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                f"<i>Showing first 20 of {len(df)} wells. Full data available in CSV export.</i>",
                styles["Normal"],
            )
        )

    doc.build(story)
    logger.info(f"Field PDF report saved to {output_path}")

    return output_path


def _generate_html_report(
    forecast: ForecastResult,
    diagnostics: FitDiagnostics | None,
    params: dict[str, float] | None,
    well_id: str | None,
) -> str:
    """Generate HTML content for well report.

    Args:
        forecast: Forecast result object
        diagnostics: Optional diagnostics result
        params: Optional model parameters
        well_id: Optional well identifier

    Returns:
        HTML string
    """

    # Build HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Reservoir Engineering Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .section { margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Reservoir Engineering Analysis Report</h1>
"""

    if well_id:
        html += (
            f'    <div class="section"><p><strong>Well ID:</strong> {well_id}</p></div>'
        )

    if params:
        html += """
    <div class="section">
        <h2>Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
    """

        for param, value in params.items():
            html += f"<tr><td>{param}</td><td>{value:.4f}</td></tr>"

        html += """
        </table>
    </div>
    """

    if diagnostics:
        html += f"""
    <div class="section">
        <h2>Diagnostics</h2>
        <div class="metric">
            <strong>RMSE:</strong> {diagnostics.rmse:.4f}
        </div>
        <div class="metric">
            <strong>MAE:</strong> {diagnostics.mae:.4f}
        </div>
        <div class="metric">
            <strong>MAPE:</strong> {diagnostics.mape:.4f}
        </div>
        <div class="metric">
            <strong>R²:</strong> {diagnostics.r_squared:.4f}
        </div>
    </div>
    """

    html += """
</body>
</html>
"""

    return html


def generate_field_summary(
    well_results: list[dict],
    output_path: str | None = None,
) -> str:
    """Generate field summary table.

    Args:
        well_results: List of well result dictionaries
        output_path: Output file path

    Returns:
        Path to generated summary file
    """
    if output_path is None:
        output_path = "field_summary.csv"

    # Convert to DataFrame
    df = pd.DataFrame(well_results)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Generated field summary: {len(well_results)} wells -> {output_path}")

    return output_path
