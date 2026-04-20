import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils.langutils import TEST_LANGS, TASK_LIST

def extract_task_metrics(results_dict, task_prefix):
    """Extract metrics for a given task prefix from results"""
    task_results = {}
    for key, value in results_dict.items():
        # Handle ARC and Hellaswag special cases
        if task_prefix.lower() in ['arc', 'hellaswag']:
            # Handle leaderboard tasks
            if f"leaderboard|{task_prefix}:challenge" in key or f"leaderboard|{task_prefix}" in key:
                task_results['eng'] = value['acc_norm'] if 'acc_norm' in value else value.get('acc', 0)
            
            # Handle community tasks
            elif f"community_{task_prefix}" in key.lower():
                lang = key.split('_')[2]
                task_results[lang] = value['acc_norm'] if 'acc_norm' in value else value.get('acc', 0)
                
            # Handle MLMM tasks
            elif f"mlmm_{task_prefix}" in key.lower():
                parts = key.split('_')
                lang = parts[2]
                task_results[lang] = value['acc_norm'] if 'acc_norm' in value else value.get('acc', 0)
                
        # Handle flores200 by taking second language
        elif task_prefix.lower() == 'flores200' and 'flores200:' in key:
            try:
                lang = key.split('-')[1].split('_')[0].lower()
                if 'chrf++' in value:
                    task_results[lang] = value['chrf++']
            except IndexError:
                continue
        # Handle regular cases
        elif task_prefix.lower() in key.lower():
            parts = key.split('_')
            lang = parts[1] if len(parts) > 1 else 'eng'
            if 'acc_norm' in value:
                task_results[lang] = value['acc_norm']
            elif 'acc_norm_token' in value:
                task_results[lang] = value['acc_norm_token']
    return task_results

def plot_task_results(task_name, results_by_model):
    """Create bar plot for a specific task across models and languages"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    x_labels = []
    values = []
    hues = []
    
    for model_name, task_results in results_by_model.items():
        for lang, score in task_results.items():
            x_labels.append(lang)
            values.append(score)
            hues.append(model_name)
    
    # Create bar plot with customizations
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    sns.barplot(x=x_labels, y=values, hue=hues)
    
    plt.title(f"{task_name} Results by Language")
    plt.xlabel("Language")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{task_name.lower()}_results.pdf", bbox_inches='tight')
    plt.close()

def main():
    # Get all result files from HPLT folder
    results_dir = "results/results/HPLT"
    model_results = {}
    
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            # Find the results JSON file
            json_files = [f for f in os.listdir(model_path) if f.startswith('results_') and f.endswith('.json')]
            if json_files:
                with open(os.path.join(model_path, json_files[0])) as f:
                    model_results[model_dir] = json.load(f)['results']

    # Process each task
    for task in TASK_LIST:
        task_results_by_model = {}
        
        for model_name, results in model_results.items():
            task_metrics = extract_task_metrics(results, task)
            if task_metrics:  # Only include if we found results
                task_results_by_model[model_name] = task_metrics
        
        if task_results_by_model:  # Only plot if we have results
            plot_task_results(task, task_results_by_model)

if __name__ == "__main__":
    main()
