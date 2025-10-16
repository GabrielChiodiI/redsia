# pip install torch torchvision
import torch, torchvision as tv
from torch import nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

num_classes = None
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# transformações de treino
tfms_treino = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# dataset de treino
ds_treino = ImageFolder("plantio_14_10_2024_treino", transform=tfms_treino)
num_classes = len(ds_treino.classes)

# dataloader de treino
dl_treino = DataLoader(ds_treino, batch_size=32, shuffle=True, num_workers=4)

# modelo
modelo = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
modelo = modelo.to(dispositivo)

# otimizador e critério (função de perda)
otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)
criterio = nn.CrossEntropyLoss()

# loop de treino
for epoca in range(10):
    modelo.train()
    for entradas, rotulos in dl_treino:
        entradas, rotulos = entradas.to(dispositivo), rotulos.to(dispositivo)
        otimizador.zero_grad()
        perda = criterio(modelo(entradas), rotulos)
        perda.backward()
        otimizador.step()

# EXPORTAR DEPOIS (com try/except)
try:
    entrada_dummy = torch.randn(1,3,224,224, device=dispositivo)
    torch.onnx.export(
        modelo, entrada_dummy, "resnet50_soja.onnx",
        input_names=["input"], output_names=["prob"],
        dynamic_axes={"input":{0:"batch"}, "prob":{0:"batch"}},
        opset_version=13
    )
except Exception as e:
    print(f"[AVISO] Falha na exportação ONNX: {e}")