# Imports
import os
import random
import argparse
import numpy as np
from tqdm import tqdm

# Sklearn Import
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# W&B Imports
import wandb

# Project Imports
from model_utilities import OpenCVXRayNN, ChestXRayNN, DenseNet121ChestXRayNN, DenseNet121OpenCVXRayNN
from dataset_utilities import OpenCVXray, ChestXRayAbnormalities

# Function: See the seed for reproducibility purposes
def set_seed(seed=42):

    # Random Seed
    random.seed(seed)

    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Seed
    np.random.seed(seed)

    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return



if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser(description='HS-Project-AI4Lungs: Model Training.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--results_dir', type=str, required=True, help='The path to the results directory.')
    parser.add_argument('--weights_dir', type=str, help='Directory for saving model weights.')  
    parser.add_argument('--history_dir', type=str, help='Directory for saving training history.')  
    parser.add_argument("--data_augmentation", action="store_true", help="Enable or disable data augmentation")
    parser.add_argument('--model_name', type=str, choices=['OpenCVXRayNN', 'ChestXRayNN', 'DenseNet121ChestXRayNN', 'DenseNet121OpenCVXRayNN'], required=True, help='Name of the model to be used') 
    parser.add_argument('--dataset_name', type=str, required=True, help='The dataset for the experiments.')  
    parser.add_argument('--channels', type=int, default=3, help="Número de canais da imagem.")  
    parser.add_argument('--height', type=int, default=64, help="Altura da imagem de entrada.") 
    parser.add_argument('--width', type=int, default=64, help="Largura da imagem de entrada.") 
    parser.add_argument('--nr_classes', type=int, default=3, help="Número de classes no modelo.")  
    parser.add_argument('--epochs', type=int, default=1, help="Número de épocas para o treinamento.") 
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote para o treinamento.")
    parser.add_argument('--base_data_path', type=str, required=True, help="Caminho base dos dados para treino e validação.") 

    args = parser.parse_args()



    # Start a new wandb run to track this script
    wandb.init(
        project="hs-projec-ai4lungs",
        
        config={
            "gpu_id":args.gpu_id,
            "seed": args.seed,
            "results_dir": args.results_dir,
            "weights_dir": args.weights_dir,
            "history_dir": args.history_dir,
            "data_augmentation": args.data_augmentation,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "channels": args.channels,
            "height": args.height,
            "width": args.width,
            "nr_classes": args.nr_classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "base_data_path": args.base_data_path
        }
    )



    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)

    # Some constants
    DATA_AUGMENTATION = args.data_augmentation
    model_name = args.model_name
    dataset_name = args.dataset_name
    results_dir = args.results_dir
    channels = args.channels
    height = args.height
    width = args.width
    nr_classes = args.nr_classes
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    

    # Prepare the conditions to receive more than one model name(s) and/or dataset name(s)
    if model_name == "OpenCVXRayNN":
        model = OpenCVXRayNN(
            channels=channels,
            height=height,
            width=width,
            nr_classes=nr_classes
        )
        print("Modelo OpenCVXRayNN inicializado.")

    elif model_name == "ChestXRayNN":
        model = ChestXRayNN(
            channels=channels,
            height=height,
            width=width,
            nr_classes=nr_classes
        )
        print("Modelo ChestXRayNN inicializado.")

    elif model_name == "DenseNet121ChestXRayNN":
        model = DenseNet121ChestXRayNN(
            channels=channels,
            height=height,
            width=width,
            nr_classes=nr_classes
        )
        print("Modelo DenseNet121ChestXRayNN inicializado.")

    elif model_name == "DenseNet121OpenCVXRayNN":
        model = DenseNet121OpenCVXRayNN(
            channels=channels,
            height=height,
            width=width,
            nr_classes=nr_classes
        )
        print("Modelo DenseNet121OpenCVXRayNN inicializado.")
    else:
        raise ValueError(f"Modelo {model_name} não reconhecido!")



    # Results and Weights
    weights_dir = args.weights_dir if args.weights_dir else os.path.join(args.results_dir, dataset_name, "weights")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    # History Files
    history_dir = args.history_dir if args.history_dir else os.path.join(args.results_dir, dataset_name, "history")
    if not os.path.isdir(history_dir):
        os.makedirs(history_dir)

    # Choose GPU
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # Mean and STD to Normalize
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Check if BCEWithLogitsLoss applies to 3 classes
    LOSS = torch.nn.CrossEntropyLoss()
    LEARNING_RATE = 0.001

    # You should pass a model to the optimizer
    OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Transforms, in case we do data augmentation
    if DATA_AUGMENTATION:
        # Train
        train_transforms = transforms.Compose([
            transforms.Resize((args.height, args.width)),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        # Validation
        val_transforms = transforms.Compose([
            transforms.Resize((args.height, args.width)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD), 
        ])
    else:
        # Train
        train_transforms = transforms.Compose([
            transforms.Resize((args.height, args.width)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        # Validation
        val_transforms = transforms.Compose([
            transforms.Resize((args.height, args.width)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]) 

    # Criar os datasets de treino e validação
    base_data_path = args.base_data_path 
    if dataset_name == "OpenCVXray":
        train_set = OpenCVXray(
            base_data_path=base_data_path,  # Caminho para os dados
            split="train",  # Divisão de treino
            transform=train_transforms  # As transformações de treino 
        )

        val_set = OpenCVXray(
            base_data_path=base_data_path,  # Caminho para os dados
            split="val",  # Divisão de validação
            transform=val_transforms  # As transformações de validação 
        )

    elif dataset_name == "ChestXRayAbnormalities":
        train_set = ChestXRayAbnormalities(
            base_data_path=base_data_path,
            split="train",
            transform=train_transforms
        )

        val_set = ChestXRayAbnormalities(
            base_data_path=base_data_path,
            split="val",
            transform=val_transforms
        )
    else:
        raise ValueError(f"Dataset {dataset_name} não reconhecido!")

    # DataLoaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Train model and save best weights on validation set
    min_train_loss = np.inf
    min_val_loss = np.inf

    # Initialise losses arrays
    train_losses = np.zeros((EPOCHS, ))
    val_losses = np.zeros_like(train_losses)

    # Initialise metrics arrays
    train_metrics = np.zeros((EPOCHS, 4))
    val_metrics = np.zeros_like(train_metrics)

    # Go through the number of Epochs
    for epoch in range(EPOCHS):
        # Epoch 
        print(f"Epoch: {epoch+1}")
        
        # Training Loop
        print("Training Phase")
        
        # Initialise lists to compute scores
        y_train_true = list()
        y_train_pred = list()

        # Running train loss
        run_train_loss = 0.0

        # Put model in training mode
        model.train()

        # Iterate through dataloader
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)

            # Find the loss and update the model parameters accordingly
            # Clear the gradients of all optimized variables
            OPTIMISER.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)

            # print(logits.shape, labels.shape)
            
            # Compute the batch loss
            loss = LOSS(logits.float(), labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            OPTIMISER.step()
            
            # Update batch losses
            run_train_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_train_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_train_pred += list(s_logits.cpu().detach().numpy())

        # Compute Average Train Loss
        avg_train_loss = run_train_loss/len(train_loader.dataset)

        # Compute Train Metrics
        train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
        train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='macro')
        train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='macro')
        train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='macro')

        # Print Statistics
        print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


        # Append values to the arrays
        # Train Loss
        train_losses[epoch] = avg_train_loss
        # Save it to directory
        if DATA_AUGMENTATION:
            np.save(
                file=os.path.join(history_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}_losses_da.npy"),
                arr=train_losses,
                allow_pickle=True
            )
        else:
            np.save(
                file=os.path.join(history_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}_losses.npy"),
                arr=train_losses,
                allow_pickle=True
            )

        # Train Metrics
        # Acc
        train_metrics[epoch, 0] = train_acc
        # Recall
        train_metrics[epoch, 1] = train_recall
        # Precision
        train_metrics[epoch, 2] = train_precision
        # F1-Score
        train_metrics[epoch, 3] = train_f1
        # Save it to directory
        if DATA_AUGMENTATION:
            np.save(
                file=os.path.join(history_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}_metrics_da.npy"),
                arr=train_metrics,
                allow_pickle=True
            )
        else:
            np.save(
                file=os.path.join(history_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}_metrics.npy"),
                arr=train_metrics,
                allow_pickle=True
            )


        # Log metrics to wandb
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1_score": train_f1
            }
        )

        # Update Variables
        # Min Training Loss
        if avg_train_loss < min_train_loss:
            print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
            min_train_loss = avg_train_loss

            # Save checkpoint
            if DATA_AUGMENTATION:
                model_path = os.path.join(weights_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}_da.pt")
            else:
                model_path = os.path.join(weights_dir, f"{model_name.lower()}_tr_{dataset_name.lower()}.pt")

            torch.save(model.state_dict(), model_path)
            print(f"Successfully saved at: {model_path}")

        # Validation Loop
        print("Validation Phase")


        # Initialise lists to compute scores
        y_val_true = list()
        y_val_pred = list()

        # Running train loss
        run_val_loss = 0.0

        # Put model in evaluation mode
        model.eval()

        # Deactivate gradients
        with torch.no_grad():

            # Iterate through dataloader
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):

                # Move data data anda model to GPU (or not)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                model = model.to(DEVICE)

                # Forward pass: compute predicted outputs by passing inputs to the model
                logits = model(images)
                
                # Compute the batch loss
                # Using CrossEntropy w/ Softmax
                # loss = LOSS(logits, labels)

                # Using BCE w/ Sigmoid
                loss = LOSS(logits, labels) # Calculo da perda de validação
                
                # Update batch losses
                run_val_loss += (loss.item() * images.size(0))

                # Concatenate lists
                y_val_true += list(labels.cpu().detach().numpy())
                
                # Using Softmax Activation
                # Apply Softmax on Logits and get the argmax to get the predicted labels
                # s_logits = torch.nn.Softmax(dim=1)(logits)
                # s_logits = torch.argmax(s_logits, dim=1)
                # y_val_pred += list(s_logits.cpu().detach().numpy())

                # Ativação Softmax e predição de classes
                s_logits = torch.nn.Softmax(dim=1)(logits) #converte as saídas em probbabilidades
                s_logits = torch.argmax(s_logits, dim=1) # determinar a classe perdida
                y_val_pred += list(s_logits.cpu().detach().numpy())

            # Compute Average Train Loss
            avg_val_loss = run_val_loss/len(val_loader.dataset)

            # Compute Training Accuracy
            val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred) 
            val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='macro')
            val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='macro')
            val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='macro')

            # Print Statistics
            print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

            # Append values to the arrays
            # Train Loss
            val_losses[epoch] = avg_val_loss

            if DATA_AUGMENTATION:
                # Save it to directory
                np.save(
                    file=os.path.join(history_dir, f"{model_name.lower()}_val_{dataset_name.lower()}_losses_da.npy"),
                    arr=val_losses,
                    allow_pickle=True
                )
            else:
                # Save it to directory
                np.save(
                    file=os.path.join(history_dir, f"{model_name.lower()}_val_{dataset_name.lower()}_losses.npy"),
                    arr=val_losses,
                    allow_pickle=True
                )

            # Train Metrics
            # Acc
            val_metrics[epoch, 0] = val_acc
            # Recall
            val_metrics[epoch, 1] = val_recall
            # Precision
            val_metrics[epoch, 2] = val_precision
            # F1-Score
            val_metrics[epoch, 3] = val_f1
            # Save it to directory
            if DATA_AUGMENTATION:
                np.save(
                    file=os.path.join(history_dir, f"{model_name.lower()}_val_{dataset_name.lower()}_metrics_da.npy"),
                    arr=val_metrics,
                    allow_pickle=True
                )
            else:
                np.save(
                file=os.path.join(history_dir, f"{model_name.lower()}_val_{dataset_name.lower()}_metrics.npy"),
                arr=val_metrics,
                allow_pickle=True
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "val_loss": avg_val_loss,            
                    "val_accuracy": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1_score": val_f1
                }
            )

            # Update Variables
            # Min validation loss and save if validation loss decreases
            if avg_val_loss < min_val_loss:
                print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
                min_val_loss = avg_val_loss

                # print("Saving best model on validation...")

                # Save checkpoint
                if DATA_AUGMENTATION:
                    model_path = os.path.join(weights_dir, f"{model_name.lower()}_val_{dataset_name.lower()}_da.pt")
                else:
                    model_path = os.path.join(weights_dir, f"{model_name.lower()}_val_{dataset_name.lower()}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Successfully saved at: {model_path}")

    # Finish W&B run
    wandb.finish()

print("Finished.")