import dash_html_components as html
import dash_core_components as dcc
from dash.development.base_component import Component

from plotly.tools import mpl_to_plotly

"""
Good reference: https://dash.plot.ly/dash-core-components
"""

UNDEFINED = Component.UNDEFINED

number_to_literal = {
    1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 
    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 
    9: 'Nine', 10: 'Ten', 11: 'Eleven', 12: 'Twelve',
    13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen', 16: 'Sixteen'
}
number_to_literal = {num: text.lower() for num, text in number_to_literal.items()}

def Container(children):
    return html.Div(children, className='ui container')

def Grid(children, className='ui stackable grid'):
    return html.Div(children, className=className)

def Row(children, **kwargs):
    return html.Div(children, className='row', **kwargs)

def MultiColumn(num_columns, children):
    assert isinstance(num_columns, int), "num_columns must be integer."
    assert num_columns >= 1 and num_columns <= 16, "Must: 1 <= num_columns <= 16"
    return html.Div(children, className=f'{number_to_literal[num_columns]} wide column')

def Markdown(md):
    return dcc.Markdown(md)

def TextField(id, value=None, placeholder=None, label=None, style=None, debounce=False):
    tf = dcc.Input(type='text', id=id, value=value, placeholder=placeholder, style=style, debounce=debounce)
    return html.Div(AddLabel(tf, label=label), className="ui fluid input")

def Slider(id, min, max, step=1, value=None, label=None):    
    return AddLabel(dcc.Slider(id, min=min, max=max, step=step, value=value), label)

def Dropdown(id, options, default_text, value):
    return html.Div([
        dcc.Input(type='hidden', value=value, id=id),
        html.I(className='dropdown icon'),
        html.Div(default_text, className='default text'),
        html.Div([
            html.Div(d['label'], className='item', **{'data-value': d['value']}) for d in options
        ], className='menu')
    ], className='ui simple dropdown item')

def AddLabel(html_input, label=None):
    """
    Convenience for adding html <label> to input field
    """
    id = html_input.id
    if label:
        return html.Div([
            html.Label(children=label, htmlFor=id),
            html_input,
        ])
    else:
        return html_input
    

def MatplotlibFigure(id, fig, size_in_pixels):
    """
    Use matplotlib figure as component (by converting into plotly format first.)
    size_in_pixels: width and height tuple
    
    IMPORTANT: Unfortunately the reponsive seems to break with this method. Use plotly's graph directly is better!
    """
    width_height_tuple_in_inches = (size_in_pixels[0]/fig.dpi, size_in_pixels[1]/fig.dpi)
    fig.set_size_inches(width_height_tuple_in_inches)
    plotly_fig = mpl_to_plotly(fig, strip_style=True)
    return dcc.Graph(id, figure=plotly_fig)