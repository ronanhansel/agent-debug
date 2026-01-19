# Task 49: Real Estate Agency Frontend

## IFE Analysis

**Task Type:** Frontend Design (CLIP similarity evaluation)

**IFE Identified:** Yes - Simulated User Visual Comparison Loop Broken + Parser Mismatch

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The task is explicitly a frontend HTML/Tailwind generation task, but the system's code-parsing layer expects a Python-marked code block matching the regex... This is incompatible with the requirement to output raw HTML wrapped in `<html>` tags."

2. **Evaluation 5 (Grade 1.0):** "The simulated user is mis-formed and does not behave like a cooperative human providing needed content; instead it returns meta-evaluation text about image differences"

### Root Cause

Two interacting defects:

1. **Parser Mismatch:** The harness expects ````(?:py|python)?\s*\n(.*?)\n```` but task requires HTML output. Parser error: "Your code snippet is invalid, because the regex pattern was not found in it"

2. **Contradictory Simulated User:** The simulated user's ground truth description contradicts the task spec:
   - Task spec: "property listings grid, and a right sidebar for filters"
   - User feedback: "desired design is minimalistic, with no property card or filters"

### Ground Truth Analysis

Looking at the actual ground truth HTML for Task 49:
```html
<body class="bg-gray-100">
  <div class="container mx-auto px-4 py-8">
    <div class="flex justify-center items-center h-64">
      <img src="..." alt="Real Estate Logo" class="h-32">
    </div>
    <div class="flex justify-center items-center h-16">
      <input type="text" placeholder="Search for properties..." ...>
    </div>
    <div class="flex justify-center items-center h-64">
      <p class="text-center text-gray-800">...</p>
    </div>
  </div>
</body>
```

The ground truth is indeed minimalistic (no property grid, no sidebar), but the task description says to include "a property listings grid, and a right sidebar for filters". This is a **task specification vs ground truth mismatch**.

### Impact

1. Agents following the task description will produce layouts with grids and sidebars
2. The ground truth lacks these elements
3. CLIP similarity will penalize agents that followed the specification

## Fix Strategy

This task has both infrastructure IFE (visual loop) AND a task/ground-truth specification mismatch.

### Fix Needed: Instruction Clarification

The task description should align with what the ground truth actually shows.
