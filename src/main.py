import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model import SimpleCNN
from model import resnet18
from training_class import TrainingClass
from AL_criterias import entropy_sampling, least_confident_sampling

def active_learning(strategy_function, strategy_name, full_train_dataset, test_dataset, model_class, configs):
    #function to run a single active learning experiment with the given strategy (AL strategy to choose the next samples to label)

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    labeled_indices = indices[:configs['NUM_INITIAL_SAMPLES']]
    unlabeled_indices = indices[configs['NUM_INITIAL_SAMPLES']:]

    accuracies = []
    labeled_set_sizes = []
    test_loader = DataLoader(test_dataset, batch_size=configs['BATCH_SIZE'], shuffle=False)

    print(f"\n\n Starting Experiment for: {strategy_name}")

    for i in range(configs['NUM_ITERATIONS']):#loop for each active learning iteration
        print(f"\n Iteration {i+1}/{configs['NUM_ITERATIONS']}")
        print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")

        labeled_subset = Subset(full_train_dataset, labeled_indices)
        unlabeled_subset = Subset(full_train_dataset, unlabeled_indices)

        labeled_loader = DataLoader(labeled_subset, batch_size=configs['BATCH_SIZE'], shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=configs['BATCH_SIZE'], shuffle=False)

        # initialize a new model and trainer for each iteration to train from scratch (looks so unefficient but try other methods like fine-tuning later)
        model = model_class()
        trainer = TrainingClass(model=model, device=configs['DEVICE'])

        trainer.train(labeled_loader, lr=configs['LEARNING_RATE'], epochs=configs['EPOCHS'])        # train the model on the current labeled dataset

        #EVALUATIONs
        accuracy = trainer.evaluate(test_loader)
        accuracies.append(accuracy)
        labeled_set_sizes.append(len(labeled_indices))

        probas = trainer.predict_probas(unlabeled_loader)#get the predicted probabilities for the unlabeled set from the trainwd model

        if strategy_function:#to try active learning strategies
            query_indices_in_unlabeled = strategy_function(probas, configs['QUERY_SIZE'])
        else: # random sampling from the unlabeled set
            query_indices_in_unlabeled = np.random.choice(range(len(unlabeled_indices)), configs['QUERY_SIZE'], replace=False)


        actual_query_indices = [unlabeled_indices[i] for i in query_indices_in_unlabeled]

        labeled_indices.extend(actual_query_indices) #extend the labeled set with the newly queried samples (Can there be duplicates? check later)
        # rnsure no duplicates in labeled_indices'
        new_indices = set(actual_query_indices) - set(labeled_indices)
        labeled_indices.extend(new_indices)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]
        #unlabeled_indices = [idx for idx in unlabeled_indices if idx not in actual_query_indices]

    return accuracies, labeled_set_sizes


if __name__ == '__main__':

    CONFIGS = {
        'NUM_INITIAL_SAMPLES': 1000,
        'QUERY_SIZE': 1000,
        'NUM_ITERATIONS': 10,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.001,
        'EPOCHS': 15,# epochs per active learning iteration
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    print(f"Running on device: {CONFIGS['DEVICE']}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    entropy_accs, set_sizes = active_learning(entropy_sampling, "Entropy Sampling", full_train_dataset, test_dataset, resnet18, CONFIGS)
    least_confident_accs, _ = active_learning(least_confident_sampling, "Least Confident Sampling", full_train_dataset, test_dataset, resnet18, CONFIGS)
    random_accs, _ = active_learnsing(None, "Random Sampling", full_train_dataset, test_dataset, resnet18, CONFIGS)
   
    print("\n\n Starting experiment for fully trained benchmark model ")
    full_train_loader = DataLoader(full_train_dataset, batch_size=CONFIGS['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIGS['BATCH_SIZE'], shuffle=False)
    
    benchmark_model = SimpleCNN()
    benchmark_trainer = TrainingClass(model=benchmark_model, device=CONFIGS['DEVICE'])
    benchmark_trainer.train(full_train_loader, lr=CONFIGS['LEARNING_RATE'], epochs=CONFIGS['EPOCHS'])
    benchmark_accuracy = benchmark_trainer.evaluate(test_loader)


    plt.figure(figsize=(12, 8))
    plt.plot(set_sizes, entropy_accs, marker='o', label='Active Learning (Entropy)')
    plt.plot(set_sizes, random_accs, marker='x', label='Random Samplisng')
    plt.plot(set_sizes, least_confident_accs, marker='v', label='ACtive Learning (Least Confident')

    
    plt.axhline(y=benchmark_accuracy, color='r', linestyle='--', label=f'Benchmark (All Data): {benchmark_accuracy:.2f}%')
    
    plt.title('Active Learning Performance Comparison')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/active_learning_comparison.png')
    print("\nPlot saved to results/active_learning_comparison.png")
    plt.show()
