import plotly.graph_objs as go
import numpy as np
from django.shortcuts import render
from numpy import load

def update_graph_case2(request):
    global fig
    username = request.GET.get('username', '')
    value = request.GET.get('value')
    if not username:
        # Render the input form located in ANOVA2/templates/ANOVA2/input_anova2.html
        return render(request, 'update graph_case2.html')
    else:
        if value == 'Cell Viability':
            v1_int = load('./data/Case2/v1_int.npy')
            v6_int = load('./data/Case2/v6_int.npy')

            fig = go.Figure([
                go.Scatter(
                    name='Control',
                    x=np.array(range(0, 168, 12)),
                    y=v1_int,
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ),
                go.Scatter(
                    name='High Ammonia-stress',
                    x=np.array(range(0, 168, 12)),
                    y=v6_int,
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True
                ),
            ])
            fig.update_layout(
                xaxis_title='time (h)',
                yaxis_title='Viability (%)',
                title='Cell Viability',
                title_x=0.5,
                hovermode="x"
            )
            # fig.show()
        elif value == 'Glucose':
            y_cont_pred_0_int = load('./data/Case2/y_cont_pred_0_int.npy')
            y_high_pred_0_int = load('./data/Case2/y_high_pred_0_int.npy')

            fig = go.Figure([
                go.Scatter(
                    name='Control',
                    x=np.array(range(0, 168, 12)),
                    y=y_cont_pred_0_int,
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ),
                go.Scatter(
                    name='High Ammonia-stress',
                    x=np.array(range(0, 168, 12)),
                    y=y_high_pred_0_int,
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True
                ),
            ])
            fig.update_layout(
                xaxis_title='time (h)',
                yaxis_title='Concentration (g/L)',
                title='Predicted Glucose Conc.',
                title_x=0.5,
                hovermode="x"
            )
        elif value == 'Lactate':
            y_cont_pred_1_int = load('./y_cont_pred_1_int.npy')
            y_high_pred_1_int = load('./y_high_pred_1_int.npy')

            fig = go.Figure([
                go.Scatter(
                    name='Control',
                    x=np.array(range(0, 168, 12)),
                    y=y_cont_pred_1_int,
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ),
                go.Scatter(
                    name='High Ammonia-stress',
                    x=np.array(range(0, 168, 12)),
                    y=y_high_pred_1_int,
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True
                ),
            ])
            fig.update_layout(
                xaxis_title='time (h)',
                yaxis_title='Concentration (g/L)',
                title='Predicted Lactate Conc.',
                title_x=0.5,
                hovermode="x"
            )

        elif value == 'Glutamine':
            y_cont_pred_2_int = load('./data/Case2/y_cont_pred_2_int.npy')
            y_high_pred_2_int = load('./data/Case2/y_high_pred_2_int.npy')

            fig = go.Figure([
                go.Scatter(
                    name='Control',
                    x=np.array(range(0, 168, 12)),
                    y=y_cont_pred_2_int,
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ),
                go.Scatter(
                    name='High Ammonia-stress',
                    x=np.array(range(0, 168, 12)),
                    y=y_high_pred_2_int,
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True
                ),
            ])
            fig.update_layout(
                xaxis_title='time (h)',
                yaxis_title='Concentration (mM)',
                title='Predicted Glutamine Conc.',
                title_x=0.5,
                hovermode="x"
            )

        elif value == 'Ammonia':
            y_cont_pred_3_int = load('./data/Case2/y_cont_pred_3_int.npy')
            y_high_pred_3_int = load('./data/Case2/y_high_pred_3_int.npy')

            fig = go.Figure([
                go.Scatter(
                    name='Control',
                    x=np.array(range(0, 168, 12)),
                    y=y_cont_pred_3_int,
                    mode='markers+lines',
                    marker=dict(color='red', size=2),
                    showlegend=True
                ),
                go.Scatter(
                    name='High Ammonia-stress',
                    x=np.array(range(0, 168, 12)),
                    y=y_high_pred_3_int,
                    mode='markers+lines',
                    marker=dict(color='blue', size=2),
                    showlegend=True
                ),
            ])
            fig.update_layout(
                xaxis_title='time (h)',
                yaxis_title='Concentration (mM)',
                title='Predicted Ammonia Conc.',
                title_x=0.5,
                hovermode="x"
            )
        fig_html = fig.to_html(full_html=False)
        return render(request, 'update_graph_case2_result.html', {'fig_html': fig_html})
