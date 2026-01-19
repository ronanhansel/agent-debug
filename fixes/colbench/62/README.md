# Task 62: Non-profit Organization (Single Page) Frontend

## IFE Analysis

**Task Type:** Frontend Design (CLIP similarity evaluation)

**IFE Identified:** Yes - Multiple Infrastructure Issues

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The run repeatedly fails due to an external code-parsing constraint that is inconsistent with the frontend task instructions. The harness demands a Python-style fenced block matching the regex..."

2. **Evaluation 5 (Grade 1.0):** "The task spec itself is underspecified for a CLIP-based exact-match frontend benchmark... Additionally, the simulated user behavior is incoherent: when the agent asked for concrete content/branding requirements, the 'user' instead responded with comparisons to a hidden ground-truth image"

### Root Cause

1. **Parser Mismatch:** Same regex issue - harness expects Python code blocks but task requires HTML output

2. **Broken Visual Loop:** Simulated user cannot see agent preview, returns generic responses

3. **Underspecified Task vs Exact-Match Evaluation:**
   - Task: "a large hero image with a call-to-action button, a scrolling storytelling layout"
   - Ground truth: Very specific layout with background image, centered text, blue Donate button
   - Many valid designs satisfy the task description but only one matches the ground truth

### Ground Truth Analysis

The ground truth HTML shows:
- Full-page background image using `style="background-image: url(...)"`
- Centered container with an inset image
- "Non-profit Organization" heading with paragraph
- Single "Donate Now" button with blue styling

### Impact

Agents following the collaborative dialogue will produce valid non-profit websites, but without pixel-perfect alignment to the hidden ground truth, CLIP scores suffer.

## Fix Strategy

This task has infrastructure IFEs (visual loop, parser) that cannot be fixed at task level.

### No Code Fix Required

The task specification is reasonable for a collaborative design task. The issues are:
1. Infrastructure: visual comparison loop broken
2. Evaluation method: CLIP similarity against single ground truth is too strict for underspecified design tasks

**Note:** Adding instruction clarifications that describe the exact ground truth layout would be "making it easy" (revealing the answer) rather than "making it fair."
