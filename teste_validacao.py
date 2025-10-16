import os, argparse, onnxruntime as ort
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True  # evita erro com imagens corrompidas

# Usa labels.txt se existir, senão pega do ImageFolder
def carregar_rotulos(diretorio_validacao):
    if os.path.exists("labels.txt"):
        with open("labels.txt") as f:
            return [l.strip() for l in f if l.strip()]
    return ImageFolder(diretorio_validacao).classes

# Cria sessão ONNX Runtime
def carregar_modelo_onnx(caminho_modelo="resnet50_soja.onnx"):
    return ort.InferenceSession(caminho_modelo, providers=["CPUExecutionProvider"])

# Cria um gráfico da matriz de confusão e salva em arquivo em arquivo de imagem
def plotar_matriz_confusao(matriz, rotulos, arquivo_saida="matriz_confusao.png", normalizar=True):
    if normalizar:
        matriz = matriz.astype("float") / matriz.sum(axis=1, keepdims=True)
        matriz = np.nan_to_num(matriz)

    plt.figure(figsize=(10, 8))
    plt.imshow(matriz, interpolation="nearest")
    plt.title("Matriz de Confusão" + (" (normalizada)" if normalizar else ""))
    plt.colorbar()
    pos = np.arange(len(rotulos))
    plt.xticks(pos, rotulos, rotation=45, ha="right")
    plt.yticks(pos, rotulos)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=150)
    plt.close()

def principal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_validacao", default="plantio_14_10_2024_validacao")
    parser.add_argument("--tamanho_lote", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--modelo_onnx", default="resnet50_soja.onnx")
    parser.add_argument("--saida_relatorio", default="relatorio_classificacao.txt")
    parser.add_argument("--saida_matriz", default="matriz_confusao.png")
    args = parser.parse_args()

    # 1) Rotulos
    nomes_classes = carregar_rotulos(args.dir_validacao)
    qtd_classes = len(nomes_classes)

    # 2) Dataset sem aumentação
    transformacoes = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds_validacao = ImageFolder(args.dir_validacao, transform=transformacoes)
    dl_validacao = DataLoader(ds_validacao, batch_size=args.tamanho_lote, shuffle=False,
                              num_workers=args.num_workers)

    # 3) Modelo ONNX
    sessao = carregar_modelo_onnx(args.modelo_onnx)
    entrada_nome = sessao.get_inputs()[0].name

    # 4) Inferência
    y_verdadeiro, y_predito = [], []
    for imagens, rotulos in dl_validacao:
        entradas = imagens.numpy()
        saidas = sessao.run(None, {entrada_nome: entradas})[0]
        preds = np.argmax(saidas, axis=1)
        y_verdadeiro.append(rotulos.numpy())
        y_predito.append(preds)

    y_verdadeiro = np.concatenate(y_verdadeiro)
    y_predito = np.concatenate(y_predito)

    # 5) Métricas
    acuracia = accuracy_score(y_verdadeiro, y_predito)
    print(f"Acurácia (validação): {acuracia:.4f}")

    relatorio = classification_report(
        y_verdadeiro, y_predito, target_names=nomes_classes, digits=4
    )
    print("\nRelatório por classe:\n")
    print(relatorio)

    # 6) Matriz de confusão
    matriz = confusion_matrix(y_verdadeiro, y_predito, labels=list(range(qtd_classes)))
    plotar_matriz_confusao(matriz, nomes_classes, args.saida_matriz, normalizar=True)
    print(f"Matriz de confusão salva em: {args.saida_matriz}")

    # 7) Salvar relatório
    with open(args.saida_relatorio, "w") as f:
        f.write(f"Acurácia: {acuracia:.4f}\n\n")
        f.write(relatorio)
    print(f"Relatório salvo em: {args.saida_relatorio}")

if __name__ == "__main__":
    principal()
