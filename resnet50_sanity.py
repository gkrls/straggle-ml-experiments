# Save as test_single.py
import torch
from torchvision import models, datasets, transforms

# Load data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    '/home/gks/datasets/imagenet/val',
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

print(f"Val dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes")
print(f"First 5 classes: {val_dataset.classes[:5]}")
print(f"Last 5 classes: {val_dataset.classes[-5:]}")

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(num_classes=1000).to(device)
model.eval()

# Test a few batches
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        if i >= 10: break  # Just test 10 batches
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Debug first batch
        if i == 0:
            print(f"\nFirst batch debug:")
            print(f"  Label indices: {labels[:8].tolist()}")
            print(f"  Predicted indices: {predicted[:8].tolist()}")
            print(f"  Max probs: {torch.softmax(outputs, dim=1).max(dim=1)[0][:8].tolist()}")

print(f'\nAccuracy on {total} images: {100 * correct / total:.2f}%')
print("Expected: ~0.1% for random init, ~76% for pretrained")

# Also test with pretrained to verify data is correct
print("\nTesting with pretrained ResNet50...")
model_pretrained = models.resnet50(pretrained=True).to(device)
model_pretrained.eval()

correct_pt = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        if i >= 10: break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_pretrained(images)
        _, predicted = torch.max(outputs, 1)
        correct_pt += (predicted == labels).sum().item()

print(f'Pretrained accuracy on {total} images: {100 * correct_pt / total:.2f}%')
print("Should be ~70-75% if data is correct")