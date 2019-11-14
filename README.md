# cse256-model-explanation
Final project for CSE-256

Team members:
- Wirawit Rueopas (A53277204)
- Saideep Reddy Pakkeer (A53269319)


## Installation
This project is built with [Dash](https://dash.plot.ly), a Python framework for building interactive web applications.

Install Dash:
https://dash.plot.ly/installation

## To Run
`python app.py`

## Development Guideline

### Dash
The root of webpage DOM is set in `app.py`:
```python
from components.example_layout import ExampleLayout
layout = ExampleLayout() # child of BaseComponent, see example_layout.py
app.layout = layout.render()
```
where `example_layout` is just a component created with `Container` (a wrapper of `Div`)

### Grid System
The project uses the grid system from [Semantic-UI](https://semantic-ui.com), with its wrapper functions in `components/utils.py`.  Basically **a webpage consists of rows, where a row has 16 columns inside it.** The root of webpage must be `Container`.

Coding a layout goes like this:
1. Add a block that fills webpage horizonally with `Row(...)`
2. Inside `...` of `Row` you can add an array of `MultiColumn(...)` (or, simply one `Div`)
3. Inside `...` of `MultiColumn` you can add web components, e.g., `Div`