import numpy as np
from scipy.integrate import simps
# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson


def extract_clean_function(text):
    """
    Extracts the first encountered Python function or class definition from `text`,
    strips away imports, markdown fences, and standalone comments, and converts any
    non-Python syntax line to a commented line to prevent syntax errors.

    Parameters
    ----------
    text : str
        The raw input containing Python code mixed with markdown or other content.

    Returns
    -------
    str
        A cleaned string containing only one valid Python function or class.
    """
    lines = text.splitlines()
    cleaned = []
    recording = False
    base_indent = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        stripped = line.strip()

        # Skip full-line comments
        if stripped.startswith("#") and not stripped.startswith("###"):
            continue

        # Remove markdown fences entirely or comment them out
        if stripped.startswith(""):
            cleaned.append("# " + stripped)
            continue

        # Skip import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue

        # Start recording when we hit the first def/class
        if stripped.startswith(("def ", "class ")) and not recording:
            recording = True
            base_indent = len(line) - len(stripped)
            cleaned.append(line)
            continue

        # If we are recording, keep lines that are part of the block
        if recording:
            # Determine current indentation
            cur_indent = len(line) - len(stripped)

            # Stop if dedented to or before base_indent while seeing new top-level code
            if stripped and cur_indent <= base_indent:
                break

            # Convert stray markdown or non-code segments into comments
            if not stripped:
                cleaned.append(line)
            elif (
                stripped.startswith(("", "---", "", "py"))
                or stripped.startswith(">")  # blockquote
                or stripped.startswith(("*", "-", "+"))  # list items
            ):
                cleaned.append("# " + stripped)
            else:
                cleaned.append(line)

    return "\n".join(cleaned)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('2.1', 4)
target = targets[0]

n=1.5062
d=3
RL=0.025e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[1]

n=1.5062
d=4
RL=0.05e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[2]

n=1.5062
d=2
RL=0.05e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[3]

n=1.5062
d=2
RL=0.05e3
R0=1
lambda_=1.064e-3
intensity_matrix = simulate_light_diffraction(n, d, RL, R0, lambda_)
max_index_flat = np.argmax(intensity_matrix)
assert (max_index_flat==0) == target
