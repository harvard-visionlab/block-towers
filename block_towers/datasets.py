'''
    Generate BlockTowers Datasets
'''
from datasets import Dataset, DatasetDict, concatenate_datasets
from fastprogress import master_bar, progress_bar
from .simulation import generate_batch_initial_positions, generate_trajectories_parallel

from pdb import set_trace

# settings where naturally get 50/50
settings1 = {
    3: dict(num_blocks=3, side_length=.40, std=.350, truncate=.75),
    4: dict(num_blocks=4, side_length=.40, std=.280, truncate=.65),
    5: dict(num_blocks=5, side_length=.40, std=.165, truncate=.65),
    6: dict(num_blocks=6, side_length=.40, std=.130, truncate=.65),
}

# fixed params, so we need to generate more large towers to get a 50/50 subset
settings2 = {
    3: dict(num_blocks=3, side_length=.40, std=.350, truncate=.60),
    4: dict(num_blocks=4, side_length=.40, std=.350, truncate=.60),
    5: dict(num_blocks=5, side_length=.40, std=.350, truncate=.60),
    6: dict(num_blocks=6, side_length=.40, std=.350, truncate=.60),
}

def generate_blocktower_dataset(settings, gen_fun, num_samples, pct_fall=.50, test_size=.20):
    data = dict()
    for num_blocks,params in settings.items():
        stable, unstable = generate_batch_initial_positions(gen_fun, 
                                                            **params, 
                                                            num_samples=num_samples, 
                                                            pct_fall=pct_fall)

        dataset_stable = Dataset.from_dict(dict(data=stable))
        dataset_unstable = Dataset.from_dict(dict(data=unstable))
        
        # Add 'label': 0 for stable, 1 for unstable
        dataset_stable = dataset_stable.map(lambda examples: {**examples, 'label': 0, 'num_blocks': num_blocks})  
        dataset_unstable = dataset_unstable.map(lambda examples: {**examples, 'label': 1, 'num_blocks': num_blocks})
    
        data[f'stack{num_blocks}_stable'] = dataset_stable.train_test_split(test_size=test_size)
        data[f'stack{num_blocks}_unstable'] = dataset_unstable.train_test_split(test_size=test_size)

    dataset = DatasetDict(data)
    
    return dataset

def generate_trajectory_datasets(datasets, gen_fun, splits=['train', 'test']):    
    mb = master_bar(datasets.items())
    new_datasets = dict()
    for config_name, dataset in mb:
        dsets = dict()
        for split in progress_bar(splits, parent=mb):
            start_positions = dataset[split]['data']
            simulations, _ = generate_trajectories_parallel(gen_fun, start_positions)
            
            dsets[split] = Dataset.from_dict(
                dict(data=simulations, label=dataset[split]['label'], num_blocks=dataset[split]['num_blocks'])
            )
            
        new_datasets[config_name] = DatasetDict(dsets)
    
    return DatasetDict(new_datasets)