# 🚗 Car Color Classifier (CNN com PyTorch)

Este projeto é um classificador de cores de carros treinado com uma rede neural convolucional (ResNet18) usando PyTorch.

O modelo foi treinado com um dataset contendo 10 mil imagens divididas por cor. Ele é capaz de prever a cor predominante de um carro a partir de uma imagem.

---

## 📁 Estrutura do Projeto

Car_dataset/
├── data/
│ ├── train/ # Imagens de treino por cor (ex: blue/, red/, etc.)
│ ├── val/ # Imagens de validação
│ └── test/ # Imagens de teste (opcional para previsão)
├── train.py # Script para treinar o modelo
├── test_image.py # Script para testar uma imagem
├── listar_imagens_test.py # Lista caminhos de imagens no diretório test/
├── car_color_model.pt # Modelo salvo após o treinamento (opcional subir)


---

## ⚙️ Como usar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/car-color-classifier.git
cd car-color-classifier
```

2. Criar ambiente virtual e instalar dependências

python -m venv venv
source venv/bin/activate    # No Linux/Mac
venv\Scripts\activate       # No Windows

pip install torch torchvision scikit-learn tqdm pillow

3. Treinar o modelo
Certifique-se de que as pastas data/train/ e data/val/ estejam organizadas corretamente.


python train.py


4. Testar uma imagem
Edite test_image.py e defina o caminho da imagem que deseja testar:


image_path = 'data/test/blue/exemplo.jpg'

Depois execute:
python test_image.py

✨ Resultado de Exemplo
🟢 Cor prevista: RED


📌 Autor
Lucas Coelho
