"""
Script de treinamento para ADDNet2D com Xception.
Usa Comet ML para logging dos experimentos.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import comet_ml
from comet_ml import Experiment, OfflineExperiment

from addnet import ADDNet2D_Xception
from dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Treinamento ADDNet2D Xception')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/tasso/FR/our_model/deepfake_dataset/processed',
                        help='Caminho para o dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay para regularização')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Número de workers do dataloader')
    parser.add_argument('--image_size', type=int, default=299,
                        help='Tamanho da imagem (299 para Xception)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device para treinamento')
    parser.add_argument('--save_dir', type=str, default='../saved_models/addnet',
                        help='Diretório para salvar modelos')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Usar backbone pré-treinada')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint para continuar treinamento')
    parser.add_argument('--project_name', type=str, default='addnet-deepfake',
                        help='Nome do projeto no Comet')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key do Comet (ou use COMET_API_KEY env var)')
    parser.add_argument('--offline', action='store_true',
                        help='Rodar em modo offline (sem API key)')
    return parser.parse_args()


def evaluate(model, dataloader, criterion, device):
    """
    Avalia o modelo no conjunto de teste.
    
    Returns:
        loss, accuracy, predictions, true_labels
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, labels in tqdm(dataloader, desc='Avaliando', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            batch_size = labels.size(0)
            
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            
            # Pondera a loss pelo tamanho do batch (último batch pode ser menor)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Treina o modelo por uma época.
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images, masks)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Cria diretórios
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Inicializa Comet ML
    if args.offline:
        experiment = OfflineExperiment(
            project_name=args.project_name,
            offline_directory="../comet_offline",
            auto_metric_logging=True,
            auto_param_logging=True,
        )
        print("Rodando em modo OFFLINE - logs salvos em ../comet_offline")
    else:
        experiment = Experiment(
            api_key=args.api_key,  # Se None, usa COMET_API_KEY env var
            project_name=args.project_name,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
    
    # Log hyperparameters
    experiment.log_parameters({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "device": str(device),
        "optimizer": "AdamW",
        "model": "ADDNet2D_Xception",
    })
    
    print(f"Comet Experiment: {experiment.get_key()}")
    
    # Dataloaders
    print("Carregando dataset...")
    train_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    experiment.log_parameter("train_samples", len(train_loader.dataset))
    experiment.log_parameter("test_samples", len(test_loader.dataset))
    
    # Modelo
    print("Criando modelo...")
    model = ADDNet2D_Xception(num_classes=2, pretrained=args.pretrained)
    model = model.to(device)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    experiment.log_parameter("total_params", total_params)
    experiment.log_parameter("trainable_params", trainable_params)
    
    # Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Resume training se especificado
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Carregando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumindo da época {start_epoch}")
    
    # Training loop
    print("\n" + "="*50)
    print("Iniciando treinamento...")
    print("="*50 + "\n")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            # Treina
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Avalia
            test_loss, test_acc, preds, labels = evaluate(
                model, test_loader, criterion, device
            )
            
            # Scheduler step
            scheduler.step(test_acc)
            
            # Log métricas para Comet
            experiment.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": test_loss,
                "val_accuracy": test_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "best_accuracy": best_acc,
            }, step=epoch, epoch=epoch)
            
            # Matriz de confusão a cada 10 épocas
            if epoch % 10 == 0:
                cm = confusion_matrix(labels, preds)
                print(f"\nMatriz de Confusão (Epoch {epoch}):")
                print(cm)
                print(classification_report(labels, preds, target_names=['Real', 'Fake']))
                
                # Log confusion matrix to Comet
                experiment.log_confusion_matrix(
                    y_true=labels,
                    y_predicted=preds,
                    labels=['Real', 'Fake'],
                    title=f"Confusion Matrix - Epoch {epoch}",
                    epoch=epoch
                )
            
            # Salva melhor modelo
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint_path = os.path.join(
                    args.save_dir, f'addnet_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }, checkpoint_path)
                print(f"★ Novo melhor modelo salvo: {best_acc:.4f}")
                
                # Log model to Comet
                experiment.log_model("addnet_best", checkpoint_path)
            
            # Salva checkpoint periódico
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    args.save_dir, f'addnet_epoch_{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, checkpoint_path)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{args.epochs-1}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
            print(f"  Best Acc:   {best_acc*100:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
    
    except Exception as e:
        print(f"\nErro durante treinamento: {e}")
        experiment.log_text(f"Error: {str(e)}")
        raise
    
    finally:
        # Salva modelo final
        final_path = os.path.join(args.save_dir, 'addnet_final.pth')
        torch.save({
            'epoch': epoch if 'epoch' in dir() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, final_path)
        print(f"\nModelo final salvo em: {final_path}")
        
        # Log final model to Comet
        experiment.log_model("addnet_final", final_path)
        experiment.log_metric("final_best_accuracy", best_acc)
        
        experiment.end()
        print("Treinamento finalizado!")


if __name__ == "__main__":
    main()
