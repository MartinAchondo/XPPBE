import os
import json
import pandas as pd


def get_max_iteration(folder_path):
    subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    iteration_numbers = [int(d.split('_')[1]) for d in subdirectories if d.startswith('iter_')]
    if not iteration_numbers:
        return None
    return max(iteration_numbers)

def find_largest_iteration(folder_path):
    Iter = get_max_iteration(folder_path)
    largest_iteration_folder = os.path.join(folder_path, f'iter_{Iter}')
    return largest_iteration_folder

def find_loss_csv(folder_path):
    largest_iteration_folder = find_largest_iteration(os.path.join(folder_path,'iterations'))
    if largest_iteration_folder:
        loss_csv_path = os.path.join(largest_iteration_folder, 'loss.csv')
        if os.path.isfile(loss_csv_path):
            return loss_csv_path
    return None

def get_last_iteration_losses(loss_csv_path):
    df = pd.read_csv(loss_csv_path)
    last_row = df.iloc[-1]
    losses = last_row[list(df.columns)]
    return losses

def read_results_json(folder_path):
    results_json_path = os.path.join(folder_path, 'results_values.json')
    if os.path.isfile(results_json_path):
        with open(results_json_path, 'r') as json_file:
            results_data = json.load(json_file)
        return results_data
    else:
        return None

def get_results_by_sim(folder_path):
    loss_csv_path = find_loss_csv(folder_path)
    if loss_csv_path:
        results = get_last_iteration_losses(loss_csv_path)
        results_data = read_results_json(folder_path)
        if results_data:
            for name in results_data:
                results[name] = float(results_data[name])
            return results
    return None


def create_df_excel(results_dir,excel_file_path=None):
    folders = ['']
    data = dict()
    for folder in folders:
        folder_path = os.path.join(results_dir,folder) if folder!= '' else results_dir
        sims = os.listdir(folder_path)
        for sim in sims:
            sim_path = os.path.join(folder_path,sim)
            result = get_results_by_sim(sim_path)
            data[sim] = result

    data_filtered = {sim: result for sim, result in data.items() if result is not None}
    df = pd.DataFrame.from_dict(data_filtered, orient='index')

    filter_list = ["Gsolv_value",
                    "Loss_XPINN",
                    "Loss_NN1",
                    "Loss_NN2",
                    "Loss_Val_NN1",
                    "Loss_Val_NN2",
                    "Loss_continuity_u",
                    "Loss_continuity_du",
                    "Loss_Residual_R1",
                    "Loss_Residual_R2",
                    "Loss_Boundary_D2",
                    "Loss_Data_K2",
                    "L2_analytic"]
    df_filtered = df.filter(filter_list)

    if not excel_file_path is None:
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='DF_complete', index=True)
            df_filtered.to_excel(writer, sheet_name='DF_Filtered', index=True)

    return df_filtered