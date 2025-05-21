import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import random
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mode='train'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_id = self.df.iloc[idx]['image_id']
            img_path = os.path.join(self.img_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            if self.mode == 'train':
                label = self.df.iloc[idx]['label']
                return image, label
            else:
                return image, img_id
        except Exception as e:
            print(f"Error loading image {img_id}: {str(e)}")
            raise

if __name__ == "__main__":
    set_seed(42)

    # Set paths
    DATA_DIR = 'soil_classification-2025'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'train_labels.csv')
    TEST_IDS_PATH = os.path.join(DATA_DIR, 'test_ids.csv')
    SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

    # Verify data paths
    for path in [TRAIN_DIR, TEST_DIR, TRAIN_LABELS_PATH, TEST_IDS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    # Load labels
    print("Loading data...")
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    test_ids = pd.read_csv(TEST_IDS_PATH)
    print("Train labels shape:", train_labels.shape)
    print("Test ids shape:", test_ids.shape)
    print("Sample train labels:\n", train_labels.head())

    # Check class balance
    print("\nClass distribution:")
    print(train_labels['soil_type'].value_counts())

    # Label encoding
    soil_types = sorted(train_labels['soil_type'].unique())
    soil2idx = {soil:i for i, soil in enumerate(soil_types)}
    idx2soil = {i:soil for soil, i in soil2idx.items()}
    train_labels['label'] = train_labels['soil_type'].map(soil2idx)

    # Enhanced Transforms and Augmentation
    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Train/Validation Split
    train_df, val_df = train_test_split(train_labels, stratify=train_labels['label'], test_size=0.15, random_state=42)
    print("\nTrain size:", len(train_df), "Validation size:", len(val_df))

    train_dataset = SoilDataset(train_df, TRAIN_DIR, transform=train_transform, mode='train')
    val_dataset = SoilDataset(val_df, TRAIN_DIR, transform=valid_transform, mode='train')
    test_dataset = SoilDataset(test_ids, TEST_DIR, transform=valid_transform, mode='test')

    BATCH_SIZE = 32

    # Set num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    NUM_CLASSES = len(soil_types)
    # Using a more powerful model
    model = timm.create_model('efficientnet_b2', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    # Training Loop with Early Stopping
    EPOCHS = 30
    best_min_f1 = 0
    patience = 7
    counter = 0

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        preds = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                preds += predicted.cpu().numpy().tolist()
                true_labels += labels.cpu().numpy().tolist()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        f1s = f1_score(true_labels, preds, average=None)
        min_f1 = f1s.min()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Min F1: {min_f1:.4f} | F1s: {f1s}")
        
        scheduler.step(min_f1)
        if min_f1 > best_min_f1:
            best_min_f1 = min_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Min F1: {min_f1:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds += outputs.argmax(1).cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()

    accuracy = accuracy_score(true_labels, preds)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=soil_types))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    # Inference and Submission
    print("\nGenerating predictions for test set...")
    model.eval()
    test_preds = []
    ids = []
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            pred_labels = outputs.argmax(1).cpu().numpy()
            test_preds.extend(pred_labels)
            ids.extend(img_ids)

    # Prepare Submission DataFrame
    pred_soil_types = [idx2soil[idx] for idx in test_preds]
    submission = pd.DataFrame({'image_id': ids, 'soil_type': pred_soil_types})
    submission = submission[['image_id', 'soil_type']]
    print("\nSample submission:")
    print(submission.head())

    # Save Submission File
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created: submission.csv") 