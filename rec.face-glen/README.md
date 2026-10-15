# 👁️ FaceVision AI - Reconhecimento Facial com MediaPipe

Um website interativo e moderno para detecção, rastreamento e reconhecimento facial em tempo real utilizando a inteligência artificial do **Google MediaPipe**.

---

## ✨ Recursos Principais

- ⚡ **Rastreamento Facial em Tempo Real (60 FPS)**: Processamento direto no navegador via WebAssembly e WebGL com MediaPipe Face Mesh.
- 🧬 **Reconhecimento Biométrico**: Extração de assinatura geométrica 3D com 468 landmarks normalizados invariantes à distância, escala e rotação.
- 👥 **Cadastro de Pessoas Fácil**: Cadastre qualquer pessoa na frente da câmera em 1 clique informando o nome. O sistema passa a identificá-la automaticamente.
- 💾 **Persistência Dupla**: Os rostos cadastrados são salvos tanto no navegador (`localStorage`) quanto no backend local em arquivo (`faces_db.json`).
- 😊 **Detecção de Expressões e Olhos**:
  - Expressão: Sorrindo 😊 vs Neutro 😐
  - Olhos: Abertos 👀 vs Piscando 😉
  - Pose da Cabeça: Centralizado, Olhando p/ Esquerda ou Direita
- 🎨 **Visualização Holográfica**:
  - Malha 3D (Face Mesh cibernética)
  - Bounding Box futurista com nome e porcentagem de similaridade
  - Pontos anatômicos destacados
  - Linha de Scanner Laser holográfica
- 📸 **Captura de Fotos**: Tire fotos instantâneas com as anotações biométricas em alta resolução com 1 clique.
- 🎛️ **Controles Intuitivos**:
  - Ajuste de sensibilidade/tolerância de reconhecimento (50% a 95%)
  - Seleção de webcam caso você tenha mais de uma
  - Modo espelho (inverter horizontalmente)
  - Botão de limpar cadastros

---

## 🚀 Como Iniciar

### Opção 1: Inicialização em 1 Clique (Windows)
Dê um duplo clique no arquivo:
```
iniciar.bat
```
O servidor local iniciará e abrirá automaticamente seu navegador em `http://localhost:8000`.

---

### Opção 2: Pelo Terminal / Linha de Comando
No diretório do projeto, execute:
```bash
python server.py
```
Em seguida, acesse no navegador:
```
http://localhost:8000
```

---

## 📖 Como Usar o Site

1. **Ligar a Câmera**: Clique no botão azul **"Ligar Câmera Agora"**. Seu navegador solicitará permissão de acesso à webcam — clique em **Permitir**.
2. **Posicione o Rosto**: Fique de frente para a câmera. Você verá a malha digital e o status "Rosto Desconhecido".
3. **Cadastrar Pessoa**:
   - Clique no botão verde **"Cadastrar Este Rosto"**.
   - Uma janela abrirá com a foto capturada do seu rosto.
   - Digite seu nome (ex: `Glen`) e clique em **"Salvar e Reconhecer"**.
4. **Pronto!** O sistema agora exibirá seu nome com a confiança biométrica (ex: `Glen (95%)`) sempre que você olhar para a câmera!

---

## 🛠️ Tecnologias Utilizadas

- **Frontend**: HTML5, CSS3 moderno (Glassmorphism, Dark Neon), JavaScript ES6+
- **Visão Computacional**: Google MediaPipe Face Mesh (`@mediapipe/face_mesh`)
- **Backend**: Python HTTP Server com API REST integrada (zero dependências externas necessárias)
- **Persistência**: `localStorage` + `faces_db.json`
