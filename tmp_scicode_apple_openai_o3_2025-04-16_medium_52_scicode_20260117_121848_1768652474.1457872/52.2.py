import numpy as np
from scipy import integrate, optimize

def chat_completion(prompt: str, max_tokens: int = 128, stop: str | None = None, model: str | None = None) -> str:
    """
    A placeholder implementation that mimics a chat completion call without
    relying on external libraries or unsupported parameters.

    Args:
        prompt (str): The user prompt.
        max_tokens (int, optional): Maximum length of the generated response. Defaults to 128.
        stop (str | None, optional): A stop sequence that, if provided, truncates the response before this sequence. Defaults to None.
        model (str | None, optional): Ignored in this placeholder but kept for API compatibility.

    Returns:
        str: A synthesized response based on the prompt.
    """
    # Simple echo-based generation for demonstration purposes.
    response = f"Response: {prompt}"

    # If a stop sequence is provided, truncate the response at the first occurrence.
    if stop and stop in response:
        response = response.split(stop)[0]

    # Enforce max_tokens limit (here, interpreted as max characters).
    return response[:max_tokens]
def extract_function_code(text):
    """
    Extract the first Python function (including its body) found in the
    supplied text string.

    Parameters
    ----------
    text : str
        A block of text that may include code, comments, markdown, etc.

    Returns
    -------
    str
        A string containing only the extracted function definition and its body.
        If no function is found, an empty string is returned.
    """
    lines = text.splitlines()
    in_function = False
    base_indent = None
    collected = []

    for line in lines:
        stripped = line.lstrip()

        # Detect the start of a function definition
        if not in_function and stripped.startswith("def "):
            in_function = True
            base_indent = len(line) - len(stripped)
            collected.append(line.rstrip())
            continue

        # If inside a function, keep collecting lines until dedent
        if in_function:
            if stripped and (len(line) - len(stripped)) <= base_indent:
                # Dedentation indicates end of the current function
                break
            collected.append(line.rstrip())

    return "\n".join(collected)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.2', 3)
target = targets[0]

y0 = [0, -1e-5]
En = 1.0
l = 1
R = np.logspace(-2,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
target = targets[1]

y0 = [0, -1e-5]
En = 1.5
l = 2
R = np.logspace(-1,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
target = targets[2]

y0 = [0, -1e-5]
En = 2.5
l = 2
R = np.logspace(1,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
