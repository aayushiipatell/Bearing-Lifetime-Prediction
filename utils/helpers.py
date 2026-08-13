"""
helpers.py
------------------
Helper functions for the Streamlit dashboard.
"""

def get_health_status(rul):
    """
    Return bearing health based on predicted RUL.
    """

    if rul >= 2000:
        return "🟢 Healthy"

    elif rul >= 1000:
        return "🟡 Moderate Wear"

    elif rul >= 500:
        return "🟠 Needs Maintenance"

    else:
        return "🔴 Critical"


def get_status_color(rul):
    """
    Return a color based on bearing condition.
    """

    if rul >= 2000:
        return "#2ECC71"   # Green

    elif rul >= 1000:
        return "#F1C40F"   # Yellow

    elif rul >= 500:
        return "#E67E22"   # Orange

    else:
        return "#E74C3C"   # Red


def format_rul(rul):
    """
    Format prediction value.
    """
    return f"{rul:.2f} Hours"