from django.shortcuts import render
import pandas as pd
import os
from numpy import load
import plotly.figure_factory as ff
from statsmodels.formula.api import ols
import statsmodels.api as sm
from django.http import HttpResponse, JsonResponse


def anova_view(request):
    # Assuming 'noise_level', 'target', and 'n_clicks'are passed as GET parameters
    anova_table = None
    default_value=0
    default_default=0
    noise_level = request.GET.get('noise_level', default_value)
    target = request.GET.get('target', default_default)
    username = request.GET.get('username', '')
    path_for_login_user = os.path.join("user_data", username)
    file_name = f'glycosylation_experiment_noise_{noise_level}.csv'
    file_path = os.path.join("user_data", username, file_name)
    if not username:
        # Render the input form located in ANOVA2/templates/ANOVA2/input_anova2.html
        return render(request, 'input_anova2.html')
    else:
        result_df = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
        exp_conditions = load(os.path.join(path_for_login_user, 'DOE.npy'), allow_pickle=True)
        exp_conditions = exp_conditions.astype(float)
        exp_conditions = {idx + 1: exp_conditions[idx][1:] for idx in range(len(exp_conditions))}
        if target == 'HM':
                # HM
            HM_data = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            HM_data = pd.merge(HM_data, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia) ', data=HM_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

        elif target == 'FA1G1':
                # FA1G1
                # target = 'FA1G1'
            data_df = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            data_df = pd.merge(data_df, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia)', data=data_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

        elif target == 'FA2G0':
                # FA2G0
                # target = 'FA2G0'
            data_df = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            data_df = pd.merge(data_df, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia)', data=data_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

        elif target == 'FA2G1':
                # FA2G1
                # target = 'FA2G1'
            data_df = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            data_df = pd.merge(data_df, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia)', data=data_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

        elif target == 'FA2G2':
                # FA2G2
                # target = 'FA2G2'
            data_df = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            data_df = pd.merge(data_df, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia)', data=data_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

        elif target == 'SIA':
                # FA2G2
                # target = 'FA2G2'
            data_df = result_df[result_df['Glycoform'] == target]
            exp_df = pd.DataFrame(exp_conditions).T
            exp_df = exp_df.reset_index()
            exp_df.columns = ['Experiment', 'Manganese', 'Galactose', 'Ammonia']
            data_df = pd.merge(data_df, exp_df, on='Experiment')
            model = ols('Distribution ~ C(Manganese) + C(Galactose) + C(Ammonia)', data=data_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
        if anova_table is not None:
            # Convert the ANOVA table to JSON
            anova_table_json = anova_table.to_json(orient='split')
            # Return the JSON response
            return JsonResponse(anova_table_json,safe=False)  # safe=False is needed when returning something other than a dict
        else:
            return JsonResponse({"error": "ANOVA analysis could not be performed. Please check your inputs."})