import pandas as pd

def find_suitable_builds(df, budget, cpu, gpu, storage):
    # convert budget to int
    budget = int(budget) if isinstance(budget, str) else budget

    filtered_df = df.copy()
    
    # filtered by CPU
    if cpu:
        filtered_df = filtered_df[filtered_df['Processor (CPU)'].str.contains(cpu, case=False)]
    
    # filtered by GPU
    if gpu:
        gpu = gpu.replace('Nvidia', 'GeForce') # GeForce is the keyword for Nvidia GPUs
        filtered_df = filtered_df[filtered_df['Video Card (GPU)'].str.contains(gpu, case=False)]
    
    # filtered by storage
    if storage:
        # Assuming only '500GB' uses GB as unit
        if 'GB' in storage:
            pass
        else:
            filtered_df['Storage Size'] = filtered_df['Storage'].str.extract(r'(\d+)TB').astype(float)
            filtered_df = filtered_df[filtered_df['Storage Size'] >= 2]

    # find index of the most expensive under budget, -1 if not found
    under_budget_builds = filtered_df[filtered_df['Price with 1 Year Standard Warranty (USD)'] <= budget]
    most_expensive_under_budget = under_budget_builds['Price with 1 Year Standard Warranty (USD)'].idxmax() \
                                  if not under_budget_builds.empty else -1

    # find index of the cheapest over budget * 1.1, -1 if not found
    over_budget_builds = filtered_df[filtered_df['Price with 1 Year Standard Warranty (USD)'] > budget*1.1]
    cheapest_over_budget = over_budget_builds['Price with 1 Year Standard Warranty (USD)'].idxmin() \
                           if not over_budget_builds.empty else -1

    return most_expensive_under_budget, cheapest_over_budget

def my_df_row_to_str(df, index, get_comment=True):
    row = df.iloc[index]
    if get_comment:
        out_str = '\n'.join(f"{index}: {value}" for index, value in row.items())
    else:
        out_str = '\n'.join(f"{index}: {value}" for index, value in row.items() if index != 'Performance Comment')
    return out_str

df = pd.read_csv("../resources/product.csv", encoding='ISO-8859-1')
requirements = [
    (2000, "AMD", "Nvidia", ""),
    (1500, "Intel", "", "minimum 2TB"),
    (3000, "AMD", "AMD", "minimum 2TB"),
    (1000, "", "", "minimum 2TB"),
    (2500, "Intel", "", ""),
    (4000, "", "Nvidia", ""),
    (1500, "", "", "minimum 2TB"),
    (3000, "Intel", "Nvidia", ""),
    (1000, "", "", ""),
    (4000, "AMD", "AMD", "minimum 2TB")
]
for index, requirement in enumerate(requirements):
    under, over = find_suitable_builds(df, requirement[0], requirement[1], requirement[2], requirement[3])
    if under != -1:
        print("{}".format(df['Name'][under]), end="")
        if over != -1:
            print(", ", end="")
    if over != -1:
        print("{}".format(df['Name'][over]), end="")
    print("")
    
    