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
def strip_to_function(text: str) -> str:
    """
    Takes a block of text that may contain markdown, import statements, comments,
    and executable code. It returns a string that contains only the lines that
    are part of a Python function or class definition, with the following rules:
        1. Lines outside of any function/class are replaced by commented-out
           versions (prepended with '#') to avoid syntax errors.
        2. Markdown code-fence lines (...) are removed.
        3. Import statements are removed.
        4. Full-line comments are removed.
        5. Inline comments are stripped, preserving the actual code segment.
    The resulting string can be executed as valid Python code.
    """
    def _remove_inline_comment(src_line: str) -> str:
        in_string = False
        quote_char = ""
        for idx, ch in enumerate(src_line):
            if ch in {'"', "'"}:
                if not in_string:
                    in_string, quote_char = True, ch
                elif quote_char == ch:
                    in_string, quote_char = False, ""
            elif ch == "#" and not in_string:
                return src_line[:idx].rstrip()
        return src_line.rstrip()

    lines = text.splitlines()
    result = []
    in_block = False
    base_indent = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Skip markdown fences entirely
        if line.strip().startswith(""):
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Detect start of a function or class definition
        if stripped.startswith(("def ", "class ")):
            in_block = True
            base_indent = indent
            cleaned = _remove_inline_comment(stripped)
            if cleaned:
                result.append(" " * indent + cleaned)
            continue

        if in_block:
            # If indentation drops below the base indentation, block has ended
            if stripped and indent <= base_indent:
                in_block = False

        if in_block:
            # Inside a valid function/class body
            if stripped.startswith(("import ", "from ")):
                # Skip import statements
                continue
            if stripped.startswith("#") or stripped == "":
                # Remove full-line comments and empty lines
                continue
            cleaned = _remove_inline_comment(line)
            if cleaned.strip():
                result.append(cleaned)
        else:
            # Outside any function/class: comment out to avoid syntax errors
            if stripped:
                result.append("# " + stripped)

    return "\n".join(result)
def extract_first_function(text: str) -> str:
    """
    Extracts and returns the first Python function definition (with its full body)
    from a block of text. The returned code will not include any imports, markdown
    fences, or top-level comments found outside of that function.

    Parameters
    ----------
    text : str
        The input text possibly containing markdown, comments, imports, and code.

    Returns
    -------
    str
        A string containing exactly one Python function (the first one found).
        If no function is found, an empty string is returned.
    """
    lines = text.splitlines()
    in_code_block = False          # Tracks markdown fences
    recording = False              # True while copying the target function
    func_indent = 0
    function_lines = []

    for raw_line in lines:
        stripped = raw_line.lstrip()

        # Toggle code-block flag for markdown fences
        if stripped.startswith(""):
            in_code_block = not in_code_block
            continue

        # Skip processing lines that are outside a  block if any exist
        # When there are no fences, everything is treated as code
        if any(s.startswith("") for s in lines) and not in_code_block:
            continue

        # Skip top-level imports and full-line comments until we start recording
        if not recording and (stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("#")):
            continue

        # Detect the beginning of a function definition
        if not recording and (stripped.startswith("def ") or stripped.startswith("async def ")):
            recording = True
            func_indent = len(raw_line) - len(stripped)
            function_lines.append(stripped.rstrip())
            continue

        # If we're recording, decide whether the line belongs to the function body
        if recording:
            current_indent = len(raw_line) - len(stripped)

            # A non-blank line with indentation less than or equal to the
            # function's indentation signals the end of the function
            if stripped and current_indent <= func_indent:
                break

            # Keep the line as part of the function (preserve original spacing)
            function_lines.append(raw_line.rstrip())

    return "\n".join(function_lines)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.4', 3)
target = targets[0]

y0 = [0, -1e-5]
Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[1]

Esearch = -0.9/np.arange(1,20,0.2)**2
R = np.logspace(-8,2.2,1000)
nmax=5
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[2]

Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0,R,l,nmax-l,Esearch)
assert np.isclose(Bnd[0], (0,-1)).any() == target.any()
