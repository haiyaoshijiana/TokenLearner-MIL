import torch
import os
import sys
import glob
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Import all functions from model.py
from model import (
    init_model, 
    train, 
    test, 
    generate_pt_files,
    save_fold_results
)

def main():
    parser = argparse.ArgumentParser(description='Train Token Learner MIL on Histopathology Images')
    # Model parameters
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [1]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--num_tokens', default=128, type=int, help='Number of learnable tokens [128]')
    
    # Training parameters
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [50]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [1e-4]')
    parser.add_argument('--sparsity_weight', default=0.1, type=float, help='Weight for attention sparsity loss [0.1]')
    
    # Regularization
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0.3, type=float, help='Token Learner dropout rate [0.3]')
    
    # Experiment settings
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, 
                        help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test]')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')

    args = parser.parse_args()
    print("Running TokenLearner-MIL with evaluation scheme:", args.eval_scheme)

    # Set GPU devices
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    # Load dataset paths
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')

    # Generate preprocessed training files
    temp_dir = generate_pt_files(args, pd.read_csv(bags_csv))
    
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('weights_token_learner', timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Save experiment configuration
    config_path = os.path.join(save_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    if args.eval_scheme == '5-fold-cv':
        run_kfold_cross_validation(args, temp_dir, save_path)
    elif args.eval_scheme == '5-fold-cv-standalone-test':
        # Implement standalone test evaluation
        print("Standalone test evaluation not implemented yet.")
    elif args.eval_scheme == '5-time-train+valid+test':
        # Implement multiple train-valid-test runs
        print("Multiple train-valid-test runs not implemented yet.")
    else:
        print(f"Unknown evaluation scheme: {args.eval_scheme}")

def run_kfold_cross_validation(args, temp_dir, save_path):
    """Run 5-fold cross-validation experiment."""
    bags_path = glob.glob(f'{temp_dir}/*.pt')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
        print(f"\n{'='*50}")
        print(f"Starting CV fold {fold+1}/5")
        print(f"{'='*50}")
        
        # Initialize model and optimizer
        milnet, criterion, optimizer, scheduler = init_model(args)
        
        # Split data for this fold
        train_path = [bags_path[i] for i in train_index]
        test_path = [bags_path[i] for i in test_index]
        
        # Track best performance
        fold_best_score = 0
        best_ac = 0
        best_auc = 0
        counter = 0
        epoch_results = []

        for epoch in range(1, args.num_epochs+1):
            # Training
            train_loss_bag = train(args, train_path, milnet, criterion, optimizer)
            
            # Testing
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
            scheduler.step()

            # Store epoch results
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss_bag,
                'test_loss': test_loss_bag,
                'accuracy': avg_score,
                'aucs': [float(auc) for auc in aucs],
                'thresholds': [float(t) for t in thresholds_optimal]
            }
            epoch_results.append(epoch_result)

            # Print progress
            print(f'\nEpoch [{epoch}/{args.num_epochs}] train loss: {train_loss_bag:.4f} test loss: {test_loss_bag:.4f}')
            print(f'Average accuracy: {avg_score:.4f}')
            for i, auc in enumerate(aucs):
                print(f'Class {i} AUC: {auc:.4f}')

            # Check if we have a new best model
            current_score = (sum(aucs) + avg_score) / 2
            if current_score > fold_best_score:
                counter = 0
                fold_best_score = current_score
                best_ac = avg_score
                best_auc = aucs
                
                # Save model
                save_name = os.path.join(save_path, f'fold_{fold}_best_model.pth')
                torch.save(milnet.state_dict(), save_name)
                print('Best model saved at:', save_name)
                print('Best thresholds:', ' '.join(f'class-{i}>{t:.4f}' for i, t in enumerate(thresholds_optimal)))
                
                # Save best results
                best_results = {
                    'accuracy': float(best_ac),
                    'aucs': [float(auc) for auc in best_auc],
                    'thresholds': [float(t) for t in thresholds_optimal],
                    'best_epoch': epoch
                }
            else:
                counter += 1
                
            # Early stopping
            if counter > args.stop_epochs:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Save all results for this fold
        save_fold_results(save_path, fold, epoch_results, best_results)
        
        # Store results in a dictionary
        fold_results.append({
            'accuracy': best_ac,
            'aucs': best_auc,
            'best_results': best_results,
            'epoch_results': epoch_results
        })
        
        print(f"\nFold {fold+1} Results:")
        print(f"Best Accuracy: {best_ac:.4f}")
        print(f"Best AUCs: {[f'{auc:.4f}' for auc in best_auc]}")

    # Calculate and save final results
    mean_ac = np.mean([res['accuracy'] for res in fold_results])
    mean_aucs = np.mean([res['aucs'] for res in fold_results], axis=0)
    std_ac = np.std([res['accuracy'] for res in fold_results])
    std_aucs = np.std([res['aucs'] for res in fold_results], axis=0)

    # Calculate mean and std of thresholds across folds
    all_thresholds = [res['best_results']['thresholds'] for res in fold_results]
    mean_thresholds = np.mean(all_thresholds, axis=0)
    std_thresholds = np.std(all_thresholds, axis=0)

    final_results = {
        'mean_accuracy': float(mean_ac),
        'std_accuracy': float(std_ac),
        'mean_aucs': [float(auc) for auc in mean_aucs],
        'std_aucs': [float(std) for std in std_aucs],
        'mean_thresholds': [float(t) for t in mean_thresholds],
        'std_thresholds': [float(t) for t in std_thresholds],
        'per_fold_thresholds': all_thresholds,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Save final results
    final_results_path = os.path.join(save_path, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "="*50)
    print("Final Cross-Validation Results:")
    print("="*50)
    print(f"Mean Accuracy: {mean_ac:.4f} ± {std_ac:.4f}")
    for i, (mean_auc, std_auc) in enumerate(zip(mean_aucs, std_aucs)):
        print(f"Class {i} Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print("\nOptimal Thresholds:")
    for i, (mean_t, std_t) in enumerate(zip(mean_thresholds, std_thresholds)):
        print(f"Class {i} Threshold: {mean_t:.4f} ± {std_t:.4f}")
    
    print(f"\nAll results have been saved to: {save_path}")

if __name__ == '__main__':
    main()