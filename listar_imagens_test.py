import os

def listar_imagens(base_dir='data/test'):
    print(f"\n📁 Listando imagens em: {base_dir}\n")
    for subdir, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho = os.path.join(subdir, file).replace('\\', '/')
                print(f"✅ Caminho: {caminho}")

if __name__ == "__main__":
    listar_imagens()
