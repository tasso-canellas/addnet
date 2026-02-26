"""
Script de treinamento para ADDNet2D_Xception_Bandpass.
Inclui BandpassLayer treinável antes do backbone Xception.
Usa Comet ML para logging dos experimentos.
"""
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import comet_ml
from comet_ml import Experiment, OfflineExperiment

from addnet_bandpass import ADDNet2D_Xception_Bandpass
from dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Treinamento ADDNet2D Xception + Bandpass')
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
    parser.add_argument('--save_dir', type=str, default='../saved_models/addnet_bandpass',
                        help='Diretório para salvar modelos')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Usar backbone pré-treinada')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint para continuar treinamento')
    parser.add_argument('--project_name', type=str, default='addnet-deepfake-bandpass',
                        help='Nome do projeto no Comet')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key do Comet (ou use COMET_API_KEY env var)')
    parser.add_argument('--offline', action='store_true',
                        help='Rodar em modo offline (sem API key)')

    # Parâmetros da BandpassLayer
    parser.add_argument('--kernel_size', type=int, default=31,
                        help='Tamanho do kernel da BandpassLayer')
    parser.add_argument('--freq_high', type=float, default=60.0,
                        help='Frequência de corte superior (raio maior na frequência)')
    parser.add_argument('--freq_low', type=float, default=10.0,
                        help='Frequência de corte inferior (raio menor na frequência)')

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


def log_bandpass_metrics(experiment, model, epoch, image_size=299):
    """
    Loga métricas da BandpassLayer no Comet: sigmas, frequências de corte.
    """
    bp = model.bandpass
    sigma_base = torch.abs(bp.sigma_base).item()
    gap = torch.abs(bp.gap).item()
    sigma_high = sigma_base + gap

    experiment.log_metrics({
        "bandpass/sigma_base": sigma_base,
        "bandpass/sigma_high": sigma_high,
        "bandpass/gap": gap,
    }, step=epoch, epoch=epoch)

    # Converte para frequências interpretáveis
    freq_high_cut = image_size / (2 * math.pi * (sigma_base + 1e-5))
    freq_low_cut = image_size / (2 * math.pi * (sigma_high + 1e-5))
    freq_gap = freq_high_cut - freq_low_cut

    experiment.log_metrics({
        "bandpass/freq_high_cut": freq_high_cut,
        "bandpass/freq_low_cut": freq_low_cut,
        "bandpass/freq_gap": freq_gap,
    }, step=epoch, epoch=epoch)


def log_bandpass_images(experiment, model, dataloader, device, epoch, num_images=8):
    """
    Loga exemplos de imagens antes e depois da BandpassLayer no Comet.
    Pega um batch do dataloader, aplica model.bandpass e loga as imagens.
    """
    model.eval()
    with torch.no_grad():
        # Pega um batch
        images, masks, labels = next(iter(dataloader))
        images = images[:num_images].to(device)
        labels = labels[:num_images]

        # Aplica bandpass
        filtered = model.bandpass(images)

        # Normaliza as imagens filtradas para [0, 1] para visualização
        # (bandpass output pode ter valores negativos)
        filtered_vis = filtered.clone()
        for i in range(filtered_vis.size(0)):
            fmin = filtered_vis[i].min()
            fmax = filtered_vis[i].max()
            if fmax - fmin > 1e-8:
                filtered_vis[i] = (filtered_vis[i] - fmin) / (fmax - fmin)
            else:
                filtered_vis[i] = filtered_vis[i] * 0

        # Cria grids de imagens
        # Desnormaliza as imagens originais (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = images_denorm.clamp(0, 1)

        # Grid de imagens originais
        grid_original = make_grid(images_denorm.cpu(), nrow=4, normalize=False, padding=2)
        # Grid de imagens filtradas
        grid_filtered = make_grid(filtered_vis.cpu(), nrow=4, normalize=False, padding=2)

        # Converte para numpy HWC para logging
        np_original = grid_original.permute(1, 2, 0).numpy()
        np_filtered = grid_filtered.permute(1, 2, 0).numpy()

        # Monta labels como string
        label_names = ['Real' if l == 0 else 'Fake' for l in labels.cpu().numpy()]
        label_str = ', '.join(label_names)

        # Loga no Comet
        experiment.log_image(
            np_original,
            name=f"bandpass/original_epoch_{epoch}",
            step=epoch,
            overwrite=False,
        )
        experiment.log_image(
            np_filtered,
            name=f"bandpass/filtered_epoch_{epoch}",
            step=epoch,
            overwrite=False,
        )

        print(f"  [LOG] Imagens bandpass logadas - Epoch {epoch} | Labels: {label_str}")

    model.train()


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
            api_key=args.api_key,
            project_name=args.project_name,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )

    # --- Conversão frequência -> sigma espacial ---
    img_size = args.image_size

    d_high = max(args.freq_high, 1.0)
    d_low = max(args.freq_low, 0.1)

    if d_low >= d_high:
        print(f"AVISO: Frequência baixa ({d_low}) >= Alta ({d_high}). Ajustando...")
        d_low = d_high / 2

    sigma_small = img_size / (2 * math.pi * d_high)
    sigma_large = img_size / (2 * math.pi * d_low)
    initial_gap = sigma_large - sigma_small

    print(f"--- Configuração do Filtro ---")
    print(f"Frequências Input: High={d_high} | Low={d_low}")
    print(f"Sigmas Calculados: Small={sigma_small:.4f} | Large={sigma_large:.4f}")
    print(f"Gap inicial: {initial_gap:.4f}")
    print(f"------------------------------")

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
        "model": "ADDNet2D_Xception_Bandpass",
        "kernel_size": args.kernel_size,
        "freq_high": d_high,
        "freq_low": d_low,
        "initial_sigma_small": sigma_small,
        "initial_sigma_large": sigma_large,
        "initial_gap": initial_gap,
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
    print("Criando modelo ADDNet2D_Xception_Bandpass...")
    model = ADDNet2D_Xception_Bandpass(
        num_classes=2,
        pretrained=args.pretrained,
        kernel_size=args.kernel_size,
        initial_sigma_low=sigma_small,
        initial_gap=initial_gap,
    )
    model = model.to(device)

    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bp_params = sum(p.numel() for p in model.bandpass.parameters())
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Parâmetros da BandpassLayer: {bp_params}")

    experiment.log_parameter("total_params", total_params)
    experiment.log_parameter("trainable_params", trainable_params)
    experiment.log_parameter("bandpass_params", bp_params)

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
    print("Iniciando treinamento ADDNet + Bandpass...")
    print("="*50 + "\n")

    try:
        for epoch in range(start_epoch, args.epochs):
            # Treina
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Avalia
            val_loss, val_acc, preds, labels = evaluate(
                model, test_loader, criterion, device
            )

            # Scheduler step
            scheduler.step(val_acc)

            # Log métricas por época
            experiment.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "best_accuracy": best_acc,
            }, step=epoch, epoch=epoch)

            # Log métricas da BandpassLayer
            log_bandpass_metrics(experiment, model, epoch, image_size=args.image_size)

            # Log imagens do bandpass a cada 10 épocas (e na primeira)
            if epoch % 10 == 0 or epoch == start_epoch:
                log_bandpass_images(experiment, model, train_loader, device, epoch)

            # Matriz de confusão a cada 10 épocas
            if epoch % 10 == 0:
                cm = confusion_matrix(labels, preds)
                print(f"\nMatriz de Confusão (Epoch {epoch}):")
                print(cm)
                print(classification_report(labels, preds, target_names=['Real', 'Fake']))

                experiment.log_confusion_matrix(
                    y_true=labels,
                    y_predicted=preds,
                    labels=['Real', 'Fake'],
                    title=f"Confusion Matrix - Epoch {epoch}",
                    epoch=epoch
                )

            # Salva melhor modelo
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(
                    args.save_dir, 'addnet_bandpass_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bandpass_sigma_base': model.bandpass.sigma_base.item(),
                    'bandpass_gap': model.bandpass.gap.item(),
                }, checkpoint_path)
                print(f"★ Novo melhor modelo salvo: {best_acc:.4f}")

                experiment.log_model("addnet_bandpass_best", checkpoint_path)

            # Salva checkpoint periódico
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    args.save_dir, f'addnet_bandpass_epoch_{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'bandpass_sigma_base': model.bandpass.sigma_base.item(),
                    'bandpass_gap': model.bandpass.gap.item(),
                }, checkpoint_path)

            # Print epoch summary
            bp = model.bandpass
            sigma_b = torch.abs(bp.sigma_base).item()
            gap_v = torch.abs(bp.gap).item()

            print(f"\nEpoch {epoch}/{args.epochs-1}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            print(f"  Best Acc:   {best_acc*100:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Bandpass σ_base={sigma_b:.4f} | gap={gap_v:.4f} | σ_high={sigma_b+gap_v:.4f}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    except Exception as e:
        print(f"\nErro durante treinamento: {e}")
        experiment.log_text(f"Error: {str(e)}")
        raise

    finally:
        # Salva modelo final
        final_path = os.path.join(args.save_dir, 'addnet_bandpass_final.pth')
        torch.save({
            'epoch': epoch if 'epoch' in dir() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'bandpass_sigma_base': model.bandpass.sigma_base.item(),
            'bandpass_gap': model.bandpass.gap.item(),
        }, final_path)
        print(f"\nModelo final salvo em: {final_path}")

        experiment.log_model("addnet_bandpass_final", final_path)
        experiment.log_metric("final_best_accuracy", best_acc)

        experiment.end()
        print("Treinamento finalizado!")


if __name__ == "__main__":
    main()
