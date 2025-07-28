# app.py
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np

# --- Lógica de la Simulación ---

def cargar_datos_brutos():
    """Carga únicamente los datos estáticos de los modelos."""
    return pd.DataFrame({
        'modelo': ['GPT-4o', 'Claude 3 Opus', 'LLaMA 3 70B', 'Gemini 1.5 Pro'],
        'fabricante': ['OpenAI', 'Anthropic', 'Meta', 'Google'],
        'precio_input_usd': [5.0, 15.0, 0.0, 7.0],
        'precio_output_usd': [15.0, 75.0, 0.0, 21.0],
        'mmlu_score': [88.7, 86.8, 82.0, 83.7],
        'open_source': [False, False, True, False]
    })

def ejecutar_simulacion(datos_brutos, elasticidad, costo_f, algoritmo):
    """Toma los datos brutos y los parámetros para devolver un DataFrame simulado."""
    df = datos_brutos.copy()
    
    df['precio_promedio'] = 0.7 * df['precio_output_usd'] + 0.3 * df['precio_input_usd']
    df['costo_marginal'] = np.where(df['open_source'], 0.75, 0.25 * df['precio_promedio'])

    costo_fijo_real = costo_f * 1_000_000
    k_margen = 1.0 if algoritmo == 'asincrónico' else 0.5
    
    with np.errstate(divide='ignore', invalid='ignore'):
        demanda = (np.log(df['mmlu_score']) / (df['precio_promedio']**elasticidad))
        demanda_os = (np.log(df.loc[df['open_source'], 'mmlu_score']) / (df.loc[df['open_source'], 'costo_marginal']**elasticidad))
        df['demanda_relativa'] = demanda
        df.loc[df['open_source'], 'demanda_relativa'] = demanda_os

    if df['demanda_relativa'].max() > 0:
        df['demanda_relativa'] = df['demanda_relativa'] / df['demanda_relativa'].max()
    
    df['margen_bruto'] = k_margen * (df['precio_promedio'] - df['costo_marginal'])
    df.loc[df['open_source'], 'margen_bruto'] = 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['umbral_rentabilidad_M_tokens'] = costo_fijo_real / df['margen_bruto']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# --- Inicialización y Layout de la Aplicación Dash ---
app = dash.Dash(__name__)
server = app.server # Necesario para Gunicorn

datos_base = cargar_datos_brutos()

app.layout = html.Div(style={'fontFamily': 'sans-serif'}, children=[
    html.H1("Análisis Microeconómico de Mercados de IA Generativa"),

    html.Div(className='main-container', style={'display': 'flex'}, children=[
        # Barra lateral para los controles
        html.Div(className='sidebar', style={'width': '25%', 'padding': '20px'}, children=[
            html.H3("Parámetros de Simulación"),
            html.Label("Elasticidad-Precio:"),
            dcc.Slider(id='elasticidad-slider', min=0.1, max=3.0, value=1.5, step=0.1, 
                       marks={i: str(i) for i in [0.5, 1, 1.5, 2, 2.5, 3]}),
            html.Br(),
            html.Label("Coste Fijo (USD Millones):"),
            dcc.Slider(id='costo-fijo-slider', min=100, max=5000, value=1000, step=100,
                       marks={i: str(i) for i in range(1000, 5001, 1000)}),
            html.Br(),
            html.Label("Tipo de Algoritmo de Precios:"),
            dcc.Dropdown(id='algoritmo-dropdown', options=['asincrónico', 'sincrónico'], value='asincrónico', clearable=False),
        ]),
        
        # Contenedor para toda la salida que se actualizará
        html.Div(id='dashboard-content', className='main-content', style={'width': '75%', 'padding': '20px'})
    ])
])

# --- Callback para la Interactividad ---
@app.callback(
    Output('dashboard-content', 'children'),
    [Input('elasticidad-slider', 'value'),
     Input('costo-fijo-slider', 'value'),
     Input('algoritmo-dropdown', 'value')]
)
def update_dashboard(elasticidad, costo_millones, algoritmo):
    df_simulado = ejecutar_simulacion(datos_base, elasticidad, costo_millones, algoritmo)
    
    fig1 = px.scatter(
        df_simulado, x='precio_promedio', y='mmlu_score', 
        color='fabricante', size='margen_bruto', text='modelo'
    )
    fig1.update_traces(textposition='top center')

    df_rentabilidad = df_simulado.dropna(subset=['umbral_rentabilidad_M_tokens'])
    fig2 = px.bar(
        df_rentabilidad, x='modelo', y='umbral_rentabilidad_M_tokens', color='fabricante'
    )
    fig2.update_layout(yaxis_type="log")

    tabla_df = df_simulado[['modelo', 'precio_promedio', 'margen_bruto', 'umbral_rentabilidad_M_tokens', 'demanda_relativa']].round(2)
    
    return [
        html.H4(f"Resultados para Algoritmo: {algoritmo.capitalize()}"),
        html.H5("Mapa de Competencia (Calidad vs. Precio)"),
        dcc.Graph(figure=fig1),
        html.H5("Volumen Mínimo para Cubrir Costes Fijos"),
        dcc.Graph(figure=fig2),
        html.H5("Métricas Clave Simuladas"),
        dash_table.DataTable(data=tabla_df.to_dict('records'), 
                                   columns=[{'name': i.replace('_', ' ').title(), 'id': i} for i in tabla_df.columns])
    ]

if __name__ == '__main__':
    app.run_server(debug=True)
