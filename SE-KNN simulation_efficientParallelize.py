#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
import time

# --- 1. LOAD AND PREP STATIC DATA ---
# We load the CSV once globally to avoid overhead

df_2019 = pd.read_csv('./Frank2019_clean_1.csv')

if 'Unnamed: 0' in df_2019.columns:
    df_2019.drop(['Unnamed: 0'], axis=1, inplace=True)

X_raw = df_2019.drop('SalePrice', axis=1)
y_raw = df_2019['SalePrice']

# Scale the X features globally
scaler = StandardScaler()
X_scale_array = scaler.fit_transform(X_raw)
X_scale = pd.DataFrame(X_scale_array, columns=X_raw.columns)


# --- 2. HELPER FUNCTIONS ---

def input_process(X_train, X_test):
    # spatial cols
    X_train_space = X_train[['X', 'Y']].to_numpy()
    X_test_space = X_test[['X', 'Y']].to_numpy()

    # non-spatial cols
    X_train_Xs = X_train.drop(['X', 'Y'], axis=1).to_numpy()
    X_test_Xs = X_test.drop(['X', 'Y'], axis=1).to_numpy()

    if X_train_Xs.shape[1] > 20:  # do PCA if non-spatial features >20
        pca = PCA(n_components=10)
        X_train_Xs = pca.fit_transform(X_train_Xs)
        X_test_Xs = pca.transform(X_test_Xs)

    return X_train_space, X_test_space, X_train_Xs, X_test_Xs

def cal_dist(X_train_space, X_test_space, X_train_Xs, X_test_Xs, ratio):
    # Distance calculation (Broadcasting)
    dist_space = np.sqrt(((X_train_space[:, :, None] - X_test_space[:, :, None].T) ** 2).sum(1))/X_train_space.shape[1]
    dist_Xs = np.sqrt(((X_train_Xs[:, :, None] - X_test_Xs[:, :, None].T) ** 2).sum(1))/X_train_Xs.shape[1]
    distance = ratio * dist_space + (1 - ratio) * dist_Xs
    return distance

def predict(y_train, X_test, distance, k):
    sorted_distance_indices = np.argsort(distance, axis=0)
    y_pred = np.zeros(X_test.shape[0])
    y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

    for row in range(X_test.shape[0]):
        knn_indices = sorted_distance_indices[:, row][:k]
        knn_dists = distance[knn_indices, row]

        with np.errstate(divide='ignore'):
            weights = 1.0 / knn_dists
        if np.isinf(weights).any():
            weights[np.isinf(weights)] = 1.0

        y_pred[row] = np.average(y_train_np[knn_indices], weights=weights)
    return y_pred

# --- 3. WORKER FUNCTION (PARALLELIZED) ---
def process_single_ratio(ratio, X_train_space, X_valid_space, X_train_Xs, X_valid_Xs, y_train, X_valid, y_valid, k_list):
    """Calculates metrics for one ratio across all K values."""
    #if ratio == 0:
    #    ratio = ratio + 0.0000001

    distance = cal_dist(X_train_space, X_valid_space, X_train_Xs, X_valid_Xs, ratio)
    ratio_results = []

    for k in k_list:
        y_pred = predict(y_train, X_valid, distance, k)
        rmse = np.sqrt(np.mean((y_valid - y_pred)**2))
        mae = metrics.mean_absolute_error(y_valid, y_pred)
        r2 = metrics.r2_score(y_valid, y_pred)
        ratio_results.append([rmse, mae, r2])

    # Find best K for this specific ratio based on MAE (index 1)
    min_index = np.argmin(ratio_results, axis=0)[1]
    best_metrics = ratio_results[min_index] # [RMSE, MAE, R2]

    return {
        'ratio': ratio,
        'mae': best_metrics[1],
        'rmse': best_metrics[0],
        'r2': best_metrics[2],
        'best_k': k_list[min_index]
    }

def run_single_replication(i, X_scale, y, ratio_list, k_list, cycle_no):

    if i % 10 == 0: print(f"Running replication No. {i}")

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_scale, y, test_size=0.2, random_state=i + cycle_no*100 )

    for obj in [X_train_valid, y_train_valid]:
        obj.reset_index(inplace=True, drop=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.2, random_state=i + cycle_no*100 )

    for obj in [X_train, X_valid, y_train, y_valid]:
        obj.reset_index(inplace=True, drop=True)

    X_train_space, X_valid_space, X_train_Xs, X_valid_Xs = input_process(X_train, X_valid)

    # You can now keep ratios parallel or serial. I'd start serial here:
    all_records_rep = []

    for ratio in ratio_list:
        res = process_single_ratio(
            ratio, X_train_space, X_valid_space,
            X_train_Xs, X_valid_Xs,
            y_train, X_valid, y_valid, k_list
        )
        all_records_rep.append({
            'Replication_ID': i,
            'Ratio': res['ratio'],
            'Best_K': res['best_k'],
            'MAE': res['mae'],
            'RMSE': res['rmse'],
            'R2': res['r2']
        })

    return all_records_rep

# --- 4. SIMULATION CONTROL ---

from joblib import Parallel, delayed

def run_simulation(num_replications, X_scale, y, ratio_list, k_list, cycle_no):
    print(f"Starting simulation with {num_replications} replications...")
    start_time = time.time()

    all_records_nested = Parallel(n_jobs=22, backend='loky')(
        delayed(run_single_replication)(i, X_scale, y, ratio_list, k_list, cycle_no)
        for i in range(num_replications)
    )

    # Flatten list of lists
    all_records = [rec for rep_list in all_records_nested for rec in rep_list]

    total_time = time.time() - start_time
    print(f"Simulation complete in {total_time/60:.2f} minutes.")

    return pd.DataFrame(all_records)



#%%
# --- 5. EXECUTE AND PLOT ---

# Parameters
#ratio_list = [0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
#ratio_list = [0.65 ,0.7, 0.75]
ratio_list = [0.1,0.3,0.5]
k_list = [x for x in range(2, 20, 1)]

# RUN THE SIMULATION
# WARNING: 1000 replications is heavy.
# Start with 10 to test, then change to 1000.
n_reps = 1000
cycle_no = 0
df_results = run_simulation(n_reps, X_scale, y_raw, ratio_list, k_list,cycle_no)


#df_results.to_csv('./Frank2019_SimulationRes_'+str(cycle_no)+'.csv', index=True)
#df_results.to_csv('./Frank2019_SimulationRes_ratio65to75_'+str(cycle_no)+'.csv', index=True)
df_results.to_csv('./Frank2019_SimulationRes_ratio01to05_'+str(cycle_no)+'.csv', index=True)

#%%

# --- 6. Concate two DF into one ---

res_9ratios = pd.read_csv('./Frank2019_SimulationRes_0.csv')
res_3ratios = pd.read_csv('./Frank2019_SimulationRes_ratio65to75_0.csv')

df_results = pd.concat([res_9ratios, res_3ratios])
#df_sorted = result.sort_values(by=[','Best_K', 'MAE'], ascending=[True, True])

#filtered_df = df_results[df_results['Ratio_Rounded'].isin(target_ratios)].copy()
# --- 7. VISUALIZATION ---

# Create the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Ratio', y='MAE', data=df_results, palette="Set3")

plt.title(f'Distribution of Best MAE per Ratio ({n_reps} Replications)')
plt.xlabel('Spatial Ratio (Lambda)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.grid(True, alpha=0.3)
plt.show()

# Optional: Print summary table
summary = df_results.groupby('Ratio')['MAE'].agg(['mean', 'std', 'min', 'max']).sort_values('mean')
print("\nSummary Statistics by Ratio:")
print(summary)
#%%
#df_results.to_csv('./Frank2019_SimulationRes_2.csv', index=True)

