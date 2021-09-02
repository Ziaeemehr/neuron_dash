import dash
import dash_table
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash_table.Format import Format, Scheme
from plotly.figure_factory import create_streamline, create_quiver
from neuron_dash.models.montbrio import Montbrio

format_float2 = Format(group=",", precision=2, scheme=Scheme.fixed)
header_background_color = "rgb(30, 30, 30)"
cell_background_color = "rgb(125, 180, 200)"
cell_background_editable = "white"
style_data = {
    "whiteSpace": "normal",
    "height": "auto",
    "border": "none",
    "border-bottom": "1px solid #ccc",
}

style_data_conditional_cell = [
    {
        "if": {"column_editable": False},
        "backgroundColor": "rgb(125, 180, 200)",
        "color": "white",
    }
]
data = pd.read_csv("montbrio.csv")
df_par = data.loc[data["category"] == "par"].reset_index()
df_current = data.loc[data["category"] == "cur"].reset_index()

app = dash.Dash(
    __name__,
    prevent_initial_callbacks=False,
    external_stylesheets=[dbc.themes.LITERA],  # BOOTSTRAP FLATLY
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1.0"}],
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.P(
                    id="title",
                    children="Montbrio model",
                    style={"text-align": "left",
                           "color": "black", "font-size": "30px",
                           #    "font-family": "Iranian Sans"
                           },
                )],  # width={'offset': 1},
            )
        ),
        dbc.Row(dbc.Col(
                [html.Img(src="assets/montbrio.svg", style={"width": "20%"})],
                width={"size": 12, "offset": 0})),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id="datatable-params",
                            columns=[dict(id=i,
                                          name=j,
                                          format=k,
                                          editable=m,
                                          type=l
                                          )
                                     for i, j, k, m, l in zip(
                                ["parameter", "unit", "value"],
                                ["parameter", "unit", "value"],
                                [Format(), Format(), format_float2],
                                [False, False, True],
                                ["text", "text", "numeric"],
                            )],
                            data=df_par.to_dict("records"),
                            sort_action="native",
                            sort_mode="single",
                            style_header={
                                "backgroundColor": header_background_color,
                                "color": "white"},
                            style_cell_conditional=[
                                {"if": {"column_id": c}, "textAlign": "center"}
                                for c in ["parameter", "unit", "value"]
                            ],
                            style_data=style_data,
                            style_data_conditional=style_data_conditional_cell,
                        )
                    ], xs=11, sm=11, md=5, lg=5, xl=5,
                ),
                dbc.Col(
                    [
                        dbc.Row(dbc.Col([dcc.Dropdown(
                            id='dropdown1',
                            clearable=False,
                            style={'textAlign': 'center'},
                            options=[
                                {'label': 'step', 'value': 1},
                                {'label': 'ramp', 'value': 2},
                                {'label': 'sin', 'value': 3}
                            ],
                            value=3)], xs=11, sm=11, md=5, lg=5, xl=5)),
                        html.Br(),
                        dbc.Row(
                            dbc.Col([
                                dash_table.DataTable(
                                    id="datatable-current",
                                    columns=[dict(id=i,
                                                  name=j,
                                                  format=k,
                                                  editable=m,
                                                  type=l
                                                  )
                                             for i, j, k, m, l in zip(
                                        ["parameter", 'unit', "value"],
                                        ["parameter", 'unit', "value"],
                                        [Format(), Format(), format_float2],
                                        [False, False, True],
                                        ["text", 'text', "numeric"],
                                    )],
                                    data=df_current.loc[df_current["step"] == 1].to_dict(
                                        "records"),
                                    sort_action="native",
                                    sort_mode="single",
                                    style_header={
                                        "backgroundColor": header_background_color,
                                        "color": "white"},
                                    style_cell_conditional=[
                                        {"if": {"column_id": c},
                                            "textAlign": "center"}
                                        for c in ["parameter", "unit", "value"]
                                    ],
                                    style_data=style_data,
                                    style_data_conditional=style_data_conditional_cell,
                                )
                            ], xs=11, sm=11, md=5, lg=5, xl=5,)),
                    ],
                )
            ]
        ),

        html.Br(),
        dbc.Row([
            dbc.Col([html.Div(
                children=dcc.Graph(
                    id="voltage-trace",
                ),
                className="card",)], xs=11, sm=11, md=11, lg=6, xl=6,),
            dbc.Col([
                dbc.Row([dbc.Col(dcc.Slider(
                    id='my-slider',
                    min=0,
                    max=10,
                    step=0.5,
                    value=3,
                )),
                    dbc.Col(html.Div(id='slider-output-container'))]),
                dbc.Row(html.Div(
                    children=dcc.Graph(
                        id="phaseplane-plot",
                    ),
                    className="card",))
            ])
        ]),

    ], fluid=True,
)


@app.callback(
    Output("datatable-current", "data"),
    [Input("dropdown1", "value")],
    prevent_initial_call=False
)
def update_current_table(value):

    idx = int(value)
    if idx == 1:
        df = df_current.loc[df_current["step"] == 1]
        return df.to_dict("records")
    elif idx == 2:
        df = df_current.loc[df_current["ramp"] == 1]
        return df.to_dict("records")
    else:
        df = df_current.loc[df_current["sin"] == 1]
        return df.to_dict("records")


@ app.callback(
    Output('voltage-trace', 'figure'),
    [Input("datatable-params", "derived_virtual_data"),
     Input("datatable-current", "derived_virtual_data")],
    [State("dropdown1", "value")],
    prevent_initial_call=True
)
def update_output(table_par, table_current, value):

    df_par = pd.DataFrame(table_par)
    df_c = pd.DataFrame(table_current)

    MB = Montbrio(df_par, df_c, value)

    if (MB.J is None):  # or (MB.frequency is None):
        raise PreventUpdate

    data = MB.simulate()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, x_title="Time")

    fig.add_trace(go.Scatter(x=data['t'], y=data['r'], mode='lines',
                             name='r'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['t'], y=data['v'], mode='lines',
                             name='v'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['t'], y=data['I'], mode='lines',
                             name='v'), row=3, col=1)
    fig.update_layout(autosize=False,
                      height=490)

    return fig


@ app.callback(
    [Output("phaseplane-plot", "figure"),
     Output('slider-output-container', 'children')],
    [Input('my-slider', 'value'),
     Input("datatable-params", "derived_virtual_data"),
     Input("datatable-current", "derived_virtual_data")
     ],
    prevent_initial_call=True
)
def update_phaseplane(I_value, table_par, table_current):

    df_par = pd.DataFrame(table_par)
    df_c = pd.DataFrame(table_current)

    # fig = go.Figure()

    MB = Montbrio(df_par, df_c, 1)
    if (MB.J is None):
        print("prevent")
        raise PreventUpdate

    x = [0.01, 2]
    y = [-2.5, 1.5]
    r, v, dr, dv = MB.vector_filed(x=x, y=y, nx=15, ny=15, I0=I_value)
    fig = create_quiver(r, v, dr, dv,
                        scale=.08,
                        arrow_scale=.05,
                        line_width=2)
    # fig = create_streamline(r, v, dr, dv)
    fig.update_layout(yaxis_range=y,
                      xaxis_range=x)
    data = MB.nullclines(x=x, I=I_value)
    fig.add_trace(go.Scatter(x=data['r'], y=data['rnull'], mode='lines',
                             name='r nullcline'))
    fig.add_trace(go.Scatter(x=data['r'], y=data['vnull1'], mode='lines',
                             name='v nullcline', marker=dict(color="black")))
    fig.add_trace(go.Scatter(x=data['r'], y=data['vnull2'], mode='lines',
                             name='v nullcline', marker=dict(color="black")))

    return fig, 'constant I amplitude : {:.1f}'.format(I_value)


if __name__ == "__main__":
    app.run_server(debug=True, port=9000)
