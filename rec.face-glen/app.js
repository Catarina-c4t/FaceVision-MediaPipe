/**
 * FaceVision AI - Reconhecimento Facial com MediaPipe
 * Módulo Principal: Gerenciamento de Câmera, Biometria, Mesh e Reconhecimento
 */

(function () {
  'use strict';

  // =========================================================================
  // Configurações e Estado Global
  // =========================================================================
  const state = {
    isCameraRunning: false,
    stream: null,
    deviceId: null,
    mirrorMode: false,
    
    // Configurações Visuais
    showMesh: true,
    showBox: true,
    showPoints: true,
    showScanner: true,
    threshold: 0.75, // 75% padrão

    // Biometria e Rostos
    registeredFaces: [],
    currentDetections: [],
    
    // Performance / FPS
    fps: 0,
    frameCount: 0,
    lastFpsUpdateTime: performance.now(),

    // Snapshot pendente para cadastro
    pendingFace: null,
    
    // Processamento
    isModelReady: false,
    isProcessingFrame: false,
    animationFrameId: null
  };

  // Pontos-chave invariantes do MediaPipe FaceMesh para assinatura biométrica
  // Índices mapeados com base na anatomia facial 3D canônica
  const BIOMETRIC_KEYPOINTS = [
    1,   // Ponta do Nariz
    2,   // Base do Nariz
    4,   // Septo Nasal
    6,   // Ponte Nasal
    10,  // Centro da Testa / Glabella
    13,  // Lábio Superior Interno
    14,  // Lábio Inferior Interno
    17,  // Base do Lábio Inferior
    33,  // Canto Externo Olho Esquerdo
    61,  // Canto Esquerdo da Boca
    70,  // Sobrancelha Esquerda Externa
    93,  // Mandíbula Esquerda Superior
    98,  // Narina Esquerda
    107, // Sobrancelha Esquerda Interna
    132, // Bochecha Esquerda
    133, // Canto Interno Olho Esquerdo
    136, // Mandíbula Esquerda Inferior
    145, // Pálpebra Inferior Esquerda
    150, // Queixo Lateral Esquerdo
    152, // Ponto Inferior do Queixo
    159, // Pálpebra Superior Esquerda
    168, // Raiz do Nariz (Entre os Olhos)
    172, // Queixo Inferior Esquerdo
    234, // Osso Zigomático Esquerdo (Cheekbone)
    263, // Canto Externo Olho Direito
    288, // Mandíbula Direita Inferior
    291, // Canto Direito da Boca
    300, // Sobrancelha Direita Externa
    323, // Mandíbula Direita Superior
    327, // Narina Direita
    336, // Sobrancelha Direita Interna
    361, // Bochecha Direita
    362, // Canto Interno Olho Direito
    365, // Mandíbula Direita Inferior
    374, // Pálpebra Inferior Direita
    378, // Queixo Lateral Direito
    386, // Pálpebra Superior Direita
    454  // Osso Zigomático Direito (Cheekbone)
  ];

  // =========================================================================
  // Elementos do DOM
  // =========================================================================
  const dom = {
    webcam: document.getElementById('webcam'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    viewportContainer: document.getElementById('viewportContainer'),
    laserScanner: document.getElementById('laserScanner'),
    cameraOffOverlay: document.getElementById('cameraOffOverlay'),
    modelLoadingOverlay: document.getElementById('modelLoadingOverlay'),
    
    // Status e Métricas
    cameraStatusBadge: document.getElementById('cameraStatusBadge'),
    cameraStatusDot: document.getElementById('cameraStatusDot'),
    cameraStatusText: document.getElementById('cameraStatusText'),
    fpsDisplay: document.getElementById('fpsDisplay'),
    faceCountDisplay: document.getElementById('faceCountDisplay'),

    // HUD Flutuante
    hudRecognized: document.getElementById('hudRecognized'),
    hudExpression: document.getElementById('hudExpression'),
    hudEyes: document.getElementById('hudEyes'),
    hudPose: document.getElementById('hudPose'),

    // Botões Toolbar
    toggleCameraBtn: document.getElementById('toggleCameraBtn'),
    camBtnIcon: document.getElementById('camBtnIcon'),
    camBtnLabel: document.getElementById('camBtnLabel'),
    startCamPromptBtn: document.getElementById('startCamPromptBtn'),
    openRegisterModalBtn: document.getElementById('openRegisterModalBtn'),
    takeSnapshotBtn: document.getElementById('takeSnapshotBtn'),
    flipCameraBtn: document.getElementById('flipCameraBtn'),

    // Painel Lateral
    cameraSelect: document.getElementById('cameraSelect'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    toggleMesh: document.getElementById('toggleMesh'),
    toggleBox: document.getElementById('toggleBox'),
    togglePoints: document.getElementById('togglePoints'),
    toggleScanner: document.getElementById('toggleScanner'),

    // Pessoas Cadastradas
    facesList: document.getElementById('facesList'),
    facesCount: document.getElementById('facesCount'),
    emptyState: document.getElementById('emptyState'),
    clearAllFacesBtn: document.getElementById('clearAllFacesBtn'),

    // Modal de Cadastro
    registerModal: document.getElementById('registerModal'),
    closeRegisterModalBtn: document.getElementById('closeRegisterModalBtn'),
    cancelRegisterBtn: document.getElementById('cancelRegisterBtn'),
    savePersonBtn: document.getElementById('savePersonBtn'),
    personNameInput: document.getElementById('personNameInput'),
    snapshotPreviewCanvas: document.getElementById('snapshotPreviewCanvas'),

    // Toasts
    toastContainer: document.getElementById('toastContainer')
  };

  const ctx = dom.overlayCanvas.getContext('2d');
  let faceMeshInstance = null;

  // =========================================================================
  // Inicialização do MediaPipe FaceMesh
  // =========================================================================
  async function initMediaPipe() {
    try {
      showModelLoading(true);

      if (typeof window.FaceMesh === 'undefined') {
        throw new Error('Biblioteca MediaPipe FaceMesh não encontrada via CDN.');
      }

      faceMeshInstance = new window.FaceMesh({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }
      });

      faceMeshInstance.setOptions({
        maxNumFaces: 4,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      faceMeshInstance.onResults(onFaceResults);

      // Pré-aquecimento rápido
      state.isModelReady = true;
      showModelLoading(false);
      showToast('MediaPipe FaceMesh carregado com sucesso!', 'success');
    } catch (err) {
      console.error('Falha ao inicializar MediaPipe:', err);
      showModelLoading(false);
      showToast('Erro ao carregar MediaPipe. Verifique a conexão com a internet.', 'error');
    }
  }

  // =========================================================================
  // Gerenciamento de Câmera
  // =========================================================================
  async function populateCameraDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      dom.cameraSelect.innerHTML = '';
      if (videoDevices.length === 0) {
        dom.cameraSelect.innerHTML = '<option value="">Nenhuma câmera detectada</option>';
        return;
      }

      videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Câmera ${index + 1}`;
        dom.cameraSelect.appendChild(option);
      });

      if (!state.deviceId && videoDevices.length > 0) {
        state.deviceId = videoDevices[0].deviceId;
      }
    } catch (err) {
      console.warn('Não foi possível listar dispositivos de vídeo:', err);
    }
  }

  async function startCamera() {
    if (state.isCameraRunning) return;

    try {
      showModelLoading(true);

      const constraints = {
        audio: false,
        video: {
          deviceId: state.deviceId ? { exact: state.deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.stream = stream;
      dom.webcam.srcObject = stream;

      await new Promise((resolve) => {
        dom.webcam.onloadedmetadata = () => {
          dom.webcam.play();
          resolve();
        };
      });

      // Atualiza tamanhos do canvas para bater exatamente com a resolução real
      syncCanvasDimensions();
      window.addEventListener('resize', syncCanvasDimensions);

      state.isCameraRunning = true;
      showModelLoading(false);
      updateCameraUI(true);

      // Inicia loop de detecção contínuo
      startProcessingLoop();

      // Atualiza lista de câmeras com os nomes reais após permissão
      await populateCameraDevices();

      showToast('Câmera conectada com sucesso!', 'success');
    } catch (err) {
      console.error('Erro ao acessar a câmera:', err);
      showModelLoading(false);
      updateCameraUI(false);
      
      let msg = 'Não foi possível acessar a câmera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Permissão negada. Clique no ícone de cadeado/câmera na barra do navegador para autorizar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'Nenhuma câmera encontrada conectada ao computador.';
      }
      showToast(msg, 'error');
    }
  }

  function stopCamera() {
    if (!state.isCameraRunning) return;

    if (state.animationFrameId) {
      cancelAnimationFrame(state.animationFrameId);
      state.animationFrameId = null;
    }

    if (state.stream) {
      state.stream.getTracks().forEach(track => track.stop());
      state.stream = null;
    }

    dom.webcam.srcObject = null;
    state.isCameraRunning = false;
    state.currentDetections = [];

    // Limpa tela
    ctx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
    updateCameraUI(false);
    resetHud();
    showToast('Câmera desligada.', 'info');
  }

  function toggleCamera() {
    if (state.isCameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  function syncCanvasDimensions() {
    if (!dom.webcam.videoWidth) return;
    dom.overlayCanvas.width = dom.webcam.videoWidth;
    dom.overlayCanvas.height = dom.webcam.videoHeight;
  }

  // =========================================================================
  // Loop de Processamento em Tempo Real
  // =========================================================================
  function startProcessingLoop() {
    async function loop() {
      if (!state.isCameraRunning) return;

      if (dom.webcam.readyState >= 2 && !state.isProcessingFrame && faceMeshInstance) {
        state.isProcessingFrame = true;
        try {
          await faceMeshInstance.send({ image: dom.webcam });
        } catch (err) {
          console.warn('Erro ao processar frame com FaceMesh:', err);
        } finally {
          state.isProcessingFrame = false;
        }
      }

      calculateFps();
      state.animationFrameId = requestAnimationFrame(loop);
    }

    state.animationFrameId = requestAnimationFrame(loop);
  }

  function calculateFps() {
    state.frameCount++;
    const now = performance.now();
    const elapsed = now - state.lastFpsUpdateTime;

    if (elapsed >= 1000) {
      state.fps = Math.round((state.frameCount * 1000) / elapsed);
      dom.fpsDisplay.textContent = state.fps;
      state.frameCount = 0;
      state.lastFpsUpdateTime = now;
    }
  }

  // =========================================================================
  // Extração de Assinatura Biométrica 3D Invariante
  // =========================================================================
  /**
   * Converte os landmarks 3D do MediaPipe em um vetor normalizado invariante
   * a translação, rotação de plano (roll) e escala de distância.
   */
  function extractBiometricDescriptor(landmarks) {
    if (!landmarks || landmarks.length < 468) return null;

    // Centro do olho esquerdo e direito
    const leftEye = {
      x: (landmarks[33].x + landmarks[133].x) / 2,
      y: (landmarks[33].y + landmarks[133].y) / 2,
      z: (landmarks[33].z + landmarks[133].z) / 2
    };

    const rightEye = {
      x: (landmarks[263].x + landmarks[362].x) / 2,
      y: (landmarks[263].y + landmarks[362].y) / 2,
      z: (landmarks[263].z + landmarks[362].z) / 2
    };

    // Distância interocular (Escala básica)
    const eyeDist = Math.hypot(
      rightEye.x - leftEye.x,
      rightEye.y - leftEye.y,
      rightEye.z - leftEye.z
    );

    if (eyeDist < 0.01) return null;

    // Ponto médio entre os olhos (Origem do sistema de coordenadas do rosto)
    const eyeCenter = {
      x: (leftEye.x + rightEye.x) / 2,
      y: (leftEye.y + rightEye.y) / 2,
      z: (leftEye.z + rightEye.z) / 2
    };

    // Ângulo de inclinação lateral da cabeça (Roll)
    const rollAngle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
    const cosA = Math.cos(-rollAngle);
    const sinA = Math.sin(-rollAngle);

    // Vetor de coordenadas canônicas normalizadas
    const normalizedPoints = [];

    for (let idx of BIOMETRIC_KEYPOINTS) {
      const pt = landmarks[idx];
      if (!pt) continue;

      // Translação para o centro dos olhos
      const dx = pt.x - eyeCenter.x;
      const dy = pt.y - eyeCenter.y;
      const dz = pt.z - eyeCenter.z;

      // Rotação 2D compensando a inclinação da cabeça
      const rotX = dx * cosA - dy * sinA;
      const rotY = dx * sinA + dy * cosA;
      const rotZ = dz;

      // Normalização pela distância interocular
      normalizedPoints.push(rotX / eyeDist, rotY / eyeDist, rotZ / eyeDist);
    }

    // Proporções anatômicas invariantes complementares
    const dist = (i1, i2) => Math.hypot(
      landmarks[i1].x - landmarks[i2].x,
      landmarks[i1].y - landmarks[i2].y,
      landmarks[i1].z - landmarks[i2].z
    ) / eyeDist;

    const anatomicalRatios = [
      dist(10, 152),  // Altura total da face (testa até queixo)
      dist(168, 1),   // Comprimento do nariz
      dist(1, 152),   // Distância do nariz ao queixo
      dist(61, 291),  // Largura da boca
      dist(234, 454), // Largura entre ossos zigomáticos (maçãs do rosto)
      dist(136, 365), // Largura da mandíbula
      dist(152, 17),  // Distância do lábio inferior ao queixo
      (dist(70, 33) + dist(300, 263)) / 2, // Distância sobrancelha-olho
      dist(1, 168) / (dist(1, 152) || 1)   // Proporção nasal / queixo
    ];

    return [...normalizedPoints, ...anatomicalRatios];
  }

  /**
   * Compara dois descritores biométricos e retorna a porcentagem de similaridade (0 a 100).
   */
  function compareBiometricDescriptors(descA, descB) {
    if (!descA || !descB || descA.length !== descB.length) return 0;

    const len = descA.length;
    let sumSqDiff = 0;
    let dot = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < len; i++) {
      const a = descA[i];
      const b = descB[i];
      const diff = a - b;
      sumSqDiff += diff * diff;

      dot += a * b;
      magA += a * a;
      magB += b * b;
    }

    // 1. Distância Euclidiana Normalizada
    const euclideanDist = Math.sqrt(sumSqDiff / len);

    // 2. Similaridade Cosseno
    const cosineSim = dot / (Math.sqrt(magA) * Math.sqrt(magB) || 1);

    // Escala empírica calibrada para o espaço biométrico canônico MediaPipe:
    // Rostos idênticos: dist ~ 0.03 a 0.09
    // Rostos diferentes: dist > 0.18 a 0.35
    const euclideanScore = Math.max(0, Math.min(1, 1 - (euclideanDist / 0.22)));
    const cosineScore = Math.max(0, Math.min(1, (cosineSim - 0.94) / 0.06));

    // Combinação ponderada
    const combinedScore = (euclideanScore * 0.65 + cosineScore * 0.35) * 100;
    return Math.round(Math.max(0, Math.min(100, combinedScore)));
  }

  // =========================================================================
  // Análise de Expressão Facial e Pose
  // =========================================================================
  function analyzeFaceAttributes(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return { smile: false, eyesOpen: true, pose: 'Desconhecida' };
    }

    // 1. Detecção de Sorriso
    // Mede a distância entre os cantos da boca relativo à distância interocular
    const mouthWidth = Math.hypot(
      landmarks[61].x - landmarks[291].x,
      landmarks[61].y - landmarks[291].y
    );
    const eyeDist = Math.hypot(
      landmarks[33].x - landmarks[263].x,
      landmarks[33].y - landmarks[263].y
    );
    const mouthCornerLift = ((landmarks[0].y - landmarks[61].y) + (landmarks[0].y - landmarks[291].y)) / 2;
    const isSmiling = (mouthWidth / (eyeDist || 1)) > 0.82 || mouthCornerLift > 0.015;

    // 2. Olhos Abertos / Piscando (Eye Aspect Ratio - EAR)
    const leftEyeHeight = Math.hypot(landmarks[159].x - landmarks[145].x, landmarks[159].y - landmarks[145].y);
    const leftEyeWidth = Math.hypot(landmarks[33].x - landmarks[133].x, landmarks[33].y - landmarks[133].y);
    const leftEAR = leftEyeHeight / (leftEyeWidth || 1);

    const rightEyeHeight = Math.hypot(landmarks[386].x - landmarks[374].x, landmarks[386].y - landmarks[374].y);
    const rightEyeWidth = Math.hypot(landmarks[263].x - landmarks[362].x, landmarks[263].y - landmarks[362].y);
    const rightEAR = rightEyeHeight / (rightEyeWidth || 1);

    const avgEAR = (leftEAR + rightEAR) / 2;
    const eyesOpen = avgEAR > 0.16;

    // 3. Pose da Cabeça (Rotação Yaw)
    const noseTip = landmarks[1];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const cheekMidX = (leftCheek.x + rightCheek.x) / 2;
    const yawOffset = (noseTip.x - cheekMidX) / (eyeDist || 1);

    let pose = 'Frontal';
    if (yawOffset < -0.25) {
      pose = state.mirrorMode ? 'Olhando Direita' : 'Olhando Esquerda';
    } else if (yawOffset > 0.25) {
      pose = state.mirrorMode ? 'Olhando Esquerda' : 'Olhando Direita';
    }

    return {
      smile: isSmiling,
      eyesOpen: eyesOpen,
      pose: pose
    };
  }

  // =========================================================================
  // Processamento dos Resultados do MediaPipe (onResults)
  // =========================================================================
  function onFaceResults(results) {
    if (!state.isCameraRunning) return;

    const width = dom.overlayCanvas.width;
    const height = dom.overlayCanvas.height;

    ctx.clearRect(0, 0, width, height);

    const detectedFaces = results.multiFaceLandmarks || [];
    dom.faceCountDisplay.textContent = detectedFaces.length;

    // Habilita ou desabilita botões que dependem de ter um rosto na tela
    dom.openRegisterModalBtn.disabled = detectedFaces.length === 0;
    dom.takeSnapshotBtn.disabled = !state.isCameraRunning;

    if (detectedFaces.length === 0) {
      resetHud();
      state.pendingFace = null;
      return;
    }

    // Salva o primeiro rosto para captura de cadastro
    const primaryFaceLandmarks = detectedFaces[0];
    const primaryDescriptor = extractBiometricDescriptor(primaryFaceLandmarks);
    if (primaryDescriptor) {
      state.pendingFace = {
        landmarks: primaryFaceLandmarks,
        descriptor: primaryDescriptor
      };
    }

    // Processa cada rosto detectado na tela
    detectedFaces.forEach((landmarks, index) => {
      // 1. Calcula Bounding Box
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      landmarks.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });

      // Adiciona uma margem suave
      const padX = (maxX - minX) * 0.12;
      const padY = (maxY - minY) * 0.15;
      const box = {
        x: Math.max(0, (minX - padX) * width),
        y: Math.max(0, (minY - padY * 1.3) * height),
        width: Math.min(width, (maxX - minX + padX * 2) * width),
        height: Math.min(height, (maxY - minY + padY * 2) * height)
      };

      // 2. Extrai Descritor e Reconhece
      const descriptor = extractBiometricDescriptor(landmarks);
      let match = {
        recognized: false,
        name: 'Desconhecido',
        confidence: 0,
        person: null
      };

      if (descriptor && state.registeredFaces.length > 0) {
        let bestScore = 0;
        let bestPerson = null;

        state.registeredFaces.forEach(person => {
          const score = compareBiometricDescriptors(descriptor, person.descriptor);
          if (score > bestScore) {
            bestScore = score;
            bestPerson = person;
          }
        });

        const thresholdPercent = Math.round(state.threshold * 100);
        if (bestScore >= thresholdPercent && bestPerson) {
          match.recognized = true;
          match.name = bestPerson.name;
          match.confidence = bestScore;
          match.person = bestPerson;
        } else {
          match.confidence = bestScore;
        }
      }

      // 3. Atributos da Face (Expressão, Olhos, Pose)
      const attrs = analyzeFaceAttributes(landmarks);

      // Atualiza o HUD para o rosto principal
      if (index === 0) {
        updateHud(match, attrs);
      }

      // 4. Desenha Elementos Visuais no Canvas
      if (state.showMesh) {
        drawFuturisticMesh(landmarks, width, height, match.recognized);
      }

      if (state.showPoints) {
        drawKeyPoints(landmarks, width, height);
      }

      if (state.showBox) {
        drawFuturisticBox(box, match, attrs);
      }
    });
  }

  // =========================================================================
  // Desenho Gráfico no Canvas (Renderização HUD Futurista)
  // =========================================================================
  function drawFuturisticMesh(landmarks, width, height, isRecognized) {
    ctx.save();
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = isRecognized ? 'rgba(16, 185, 129, 0.45)' : 'rgba(6, 182, 212, 0.35)';

    // Conexões de malha em grade triangular leve
    const TRIANGLE_PAIRS = [
      [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356],
      [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379],
      [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149],
      [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234],
      [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
      // Nariz
      [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2],
      // Boca
      [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270],
      [270, 409], [409, 291], [291, 375], [375, 321], [321, 405], [405, 314], [314, 17],
      [17, 84], [84, 181], [181, 91], [91, 146], [146, 61]
    ];

    ctx.beginPath();
    TRIANGLE_PAIRS.forEach(([p1, p2]) => {
      if (landmarks[p1] && landmarks[p2]) {
        ctx.moveTo(landmarks[p1].x * width, landmarks[p1].y * height);
        ctx.lineTo(landmarks[p2].x * width, landmarks[p2].y * height);
      }
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawKeyPoints(landmarks, width, height) {
    ctx.save();
    ctx.fillStyle = '#06b6d4';

    // Olhos e sobrancelhas destacados
    const highlightIndices = [33, 133, 159, 145, 263, 362, 386, 374, 1, 61, 291, 152];
    highlightIndices.forEach(idx => {
      const pt = landmarks[idx];
      if (pt) {
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    ctx.restore();
  }

  function drawFuturisticBox(box, match, attrs) {
    const { x, y, width, height } = box;
    const cornerLength = Math.min(28, width * 0.22);
    const color = match.recognized ? '#10b981' : '#f43f5e';
    const bgBadge = match.recognized ? 'rgba(16, 185, 129, 0.9)' : 'rgba(244, 63, 94, 0.9)';

    ctx.save();

    // Contorno suave semi-transparente
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(x, y, width, height);

    // Cantos futuristas reforçados (Bracketing Corners)
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = color;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + cornerLength);
    ctx.lineTo(x, y);
    ctx.lineTo(x + cornerLength, y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + width - cornerLength, y);
    ctx.lineTo(x + width, y);
    ctx.lineTo(x + width, y + cornerLength);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + height - cornerLength);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + cornerLength, y + height);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + width - cornerLength, y + height);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + width, y + height - cornerLength);
    ctx.stroke();

    // Tag Superior com Nome e Confiança
    const nameText = match.recognized
      ? `${match.name} (${match.confidence}%)`
      : `Desconhecido (${match.confidence}%)`;

    ctx.font = 'bold 14px "Plus Jakarta Sans", sans-serif';
    const textMetrics = ctx.measureText(nameText);
    const badgeW = textMetrics.width + 24;
    const badgeH = 26;
    const badgeY = Math.max(10, y - badgeH - 6);

    ctx.fillStyle = bgBadge;
    roundRect(ctx, x, badgeY, badgeW, badgeH, 6);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(nameText, x + 12, badgeY + 18);

    // Tag Inferior com Expressão
    const exprText = `${attrs.smile ? 'Sorrindo 😊' : 'Sério 😐'} • ${attrs.pose}`;
    ctx.font = '11px "Plus Jakarta Sans", sans-serif';
    const exprMetrics = ctx.measureText(exprText);
    const exprW = exprMetrics.width + 16;
    const exprH = 20;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    roundRect(ctx, x, y + height + 6, exprW, exprH, 4);
    ctx.fill();

    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(exprText, x + 8, y + height + 20);

    ctx.restore();
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  // =========================================================================
  // HUD e Interface
  // =========================================================================
  function updateHud(match, attrs) {
    if (match.recognized) {
      dom.hudRecognized.className = 'hud-item recognized-active';
      dom.hudRecognized.querySelector('.hud-text').textContent = `✅ Reconhecido: ${match.name} (${match.confidence}%)`;
    } else {
      dom.hudRecognized.className = 'hud-item unknown-active';
      dom.hudRecognized.querySelector('.hud-text').textContent = `👤 Rosto Desconhecido (${match.confidence}%)`;
    }

    dom.hudExpression.querySelector('.hud-text').textContent = `Expressão: ${attrs.smile ? 'Sorrindo 😊' : 'Sério 😐'}`;
    dom.hudEyes.querySelector('.hud-text').textContent = `Olhos: ${attrs.eyesOpen ? 'Abertos 👀' : 'Piscando 😉'}`;
    dom.hudPose.querySelector('.hud-text').textContent = `Pose: ${attrs.pose}`;
  }

  function resetHud() {
    dom.hudRecognized.className = 'hud-item';
    dom.hudRecognized.querySelector('.hud-text').textContent = state.isCameraRunning
      ? 'Aguardando rosto...'
      : 'Câmera Desligada';

    dom.hudExpression.querySelector('.hud-text').textContent = 'Expressão: --';
    dom.hudEyes.querySelector('.hud-text').textContent = 'Olhos: --';
    dom.hudPose.querySelector('.hud-text').textContent = 'Pose: --';
  }

  function updateCameraUI(isActive) {
    if (isActive) {
      dom.cameraOffOverlay.classList.add('hidden');
      dom.cameraStatusDot.className = 'status-dot dot-on';
      dom.cameraStatusText.textContent = 'Câmera Ativa';
      dom.camBtnLabel.textContent = 'Desligar Câmera';
      dom.camBtnIcon.innerHTML = '<rect x="6" y="6" width="12" height="12" fill="currentColor"/>';
      dom.toggleCameraBtn.classList.replace('btn-primary', 'btn-secondary');

      if (state.showScanner) {
        dom.laserScanner.classList.add('active');
      }
    } else {
      dom.cameraOffOverlay.classList.remove('hidden');
      dom.cameraStatusDot.className = 'status-dot dot-off';
      dom.cameraStatusText.textContent = 'Câmera Desligada';
      dom.camBtnLabel.textContent = 'Ligar Câmera';
      dom.camBtnIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
      dom.toggleCameraBtn.classList.replace('btn-secondary', 'btn-primary');

      dom.laserScanner.classList.remove('active');
      dom.fpsDisplay.textContent = '0';
      dom.faceCountDisplay.textContent = '0';
      dom.openRegisterModalBtn.disabled = true;
      dom.takeSnapshotBtn.disabled = true;
    }
  }

  function showModelLoading(show) {
    if (show) {
      dom.modelLoadingOverlay.classList.add('active');
    } else {
      dom.modelLoadingOverlay.classList.remove('active');
    }
  }

  // =========================================================================
  // Gerenciamento e Cadastro de Rostos
  // =========================================================================
  function openRegisterModal() {
    if (!state.isCameraRunning || !state.pendingFace) {
      showToast('Posicione seu rosto na frente da câmera primeiro!', 'error');
      return;
    }

    const landmarks = state.pendingFace.landmarks;
    const vidW = dom.webcam.videoWidth;
    const vidH = dom.webcam.videoHeight;

    // Calcula área da face para gerar miniatura quadrada
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    landmarks.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const boxW = (maxX - minX) * vidW;
    const boxH = (maxY - minY) * vidH;
    const centerX = ((minX + maxX) / 2) * vidW;
    const centerY = ((minY + maxY) / 2) * vidH;
    const cropSize = Math.max(boxW, boxH) * 1.4;

    const sourceX = Math.max(0, centerX - cropSize / 2);
    const sourceY = Math.max(0, centerY - cropSize / 2);

    // Desenha miniatura no canvas de preview
    const prevCanvas = dom.snapshotPreviewCanvas;
    const prevCtx = prevCanvas.getContext('2d');
    prevCtx.clearRect(0, 0, prevCanvas.width, prevCanvas.height);

    if (state.mirrorMode) {
      prevCtx.save();
      prevCtx.scale(-1, 1);
      prevCtx.drawImage(
        dom.webcam,
        sourceX, sourceY, cropSize, cropSize,
        -prevCanvas.width, 0, prevCanvas.width, prevCanvas.height
      );
      prevCtx.restore();
    } else {
      prevCtx.drawImage(
        dom.webcam,
        sourceX, sourceY, cropSize, cropSize,
        0, 0, prevCanvas.width, prevCanvas.height
      );
    }

    dom.personNameInput.value = '';
    dom.registerModal.classList.add('active');
    setTimeout(() => dom.personNameInput.focus(), 150);
  }

  function closeRegisterModal() {
    dom.registerModal.classList.remove('active');
    dom.personNameInput.value = '';
  }

  async function saveRegisteredPerson() {
    const name = dom.personNameInput.value.trim();
    if (!name) {
      showToast('Por favor, informe o nome da pessoa.', 'error');
      dom.personNameInput.focus();
      return;
    }

    if (!state.pendingFace || !state.pendingFace.descriptor) {
      showToast('Rosto perdido. Tente novamente na câmera.', 'error');
      closeRegisterModal();
      return;
    }

    const avatarDataUrl = dom.snapshotPreviewCanvas.toDataURL('image/jpeg', 0.85);

    const newPerson = {
      id: 'face_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
      name: name,
      createdAt: new Date().toLocaleString('pt-BR'),
      avatar: avatarDataUrl,
      descriptor: state.pendingFace.descriptor
    };

    state.registeredFaces.push(newPerson);
    saveFacesToStorage();
    renderFacesList();

    // Sincroniza com servidor Python em segundo plano
    syncFaceWithServer(newPerson);

    closeRegisterModal();
    showToast(`Rosto de "${name}" cadastrado com sucesso!`, 'success');
  }

  function deletePerson(id) {
    const person = state.registeredFaces.find(p => p.id === id);
    const name = person ? person.name : 'Pessoa';

    state.registeredFaces = state.registeredFaces.filter(p => p.id !== id);
    saveFacesToStorage();
    renderFacesList();

    // Deleta no backend Python
    fetch(`/api/faces?id=${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => {});

    showToast(`Cadastro de "${name}" removido.`, 'info');
  }

  function clearAllFaces() {
    if (state.registeredFaces.length === 0) return;

    if (confirm('Deseja realmente apagar todos os rostos cadastrados?')) {
      state.registeredFaces = [];
      saveFacesToStorage();
      renderFacesList();

      fetch('/api/faces', { method: 'DELETE' }).catch(() => {});
      showToast('Todos os cadastros foram removidos.', 'info');
    }
  }

  function renderFacesList() {
    dom.facesCount.textContent = state.registeredFaces.length;

    if (state.registeredFaces.length === 0) {
      dom.facesList.innerHTML = '';
      dom.facesList.appendChild(dom.emptyState);
      dom.emptyState.style.display = 'flex';
      return;
    }

    dom.emptyState.style.display = 'none';
    dom.facesList.innerHTML = '';

    state.registeredFaces.forEach(person => {
      const item = document.createElement('div');
      item.className = 'face-item';

      item.innerHTML = `
        <div class="face-profile">
          <img class="face-avatar" src="${person.avatar}" alt="${person.name}">
          <div class="face-details">
            <h4>${escapeHtml(person.name)}</h4>
            <span>Cadastrado em ${person.createdAt}</span>
          </div>
        </div>
        <button class="face-delete-btn" title="Excluir cadastro" data-id="${person.id}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      `;

      item.querySelector('.face-delete-btn').addEventListener('click', () => {
        deletePerson(person.id);
      });

      dom.facesList.appendChild(item);
    });
  }

  // =========================================================================
  // Persistência (LocalStorage + Servidor Local)
  // =========================================================================
  function saveFacesToStorage() {
    try {
      localStorage.setItem('facescan_profiles', JSON.stringify(state.registeredFaces));
    } catch (e) {
      console.warn('Erro ao salvar no localStorage:', e);
    }
  }

  async function loadFaces() {
    // 1. Carrega do localStorage primeiro (imediato)
    try {
      const local = localStorage.getItem('facescan_profiles');
      if (local) {
        state.registeredFaces = JSON.parse(local);
        renderFacesList();
      }
    } catch (e) {
      console.warn('Falha ao carregar do localStorage:', e);
    }

    // 2. Busca do servidor Python local e mescla
    try {
      const res = await fetch('/api/faces');
      if (res.ok) {
        const serverFaces = await res.json();
        if (Array.isArray(serverFaces) && serverFaces.length > 0) {
          // Mescla sem duplicar IDs
          const idMap = new Map();
          state.registeredFaces.forEach(f => idMap.set(f.id, f));
          serverFaces.forEach(f => idMap.set(f.id, f));
          state.registeredFaces = Array.from(idMap.values());
          saveFacesToStorage();
          renderFacesList();
        }
      }
    } catch (e) {
      // Se não houver servidor REST, o LocalStorage garante o funcionamento 100% autônomo
    }
  }

  async function syncFaceWithServer(person) {
    try {
      await fetch('/api/faces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(person)
      });
    } catch (err) {
      console.warn('Servidor offline para sync; mantido em cache local.');
    }
  }

  // =========================================================================
  // Captura de Snapshot / Foto
  // =========================================================================
  function takeSnapshot() {
    if (!state.isCameraRunning) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = dom.webcam.videoWidth;
    exportCanvas.height = dom.webcam.videoHeight;
    const eCtx = exportCanvas.getContext('2d');

    // Desenha o vídeo
    if (state.mirrorMode) {
      eCtx.save();
      eCtx.scale(-1, 1);
      eCtx.drawImage(dom.webcam, -exportCanvas.width, 0, exportCanvas.width, exportCanvas.height);
      eCtx.restore();
    } else {
      eCtx.drawImage(dom.webcam, 0, 0, exportCanvas.width, exportCanvas.height);
    }

    // Desenha as sobreposições de mesh/caixas do overlay canvas
    eCtx.drawImage(dom.overlayCanvas, 0, 0);

    // Marca d'água discreta
    eCtx.font = 'bold 16px "Plus Jakarta Sans", sans-serif';
    eCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    eCtx.fillText(`FaceVision AI • ${new Date().toLocaleTimeString('pt-BR')}`, 20, exportCanvas.height - 20);

    // Download do arquivo
    const link = document.createElement('a');
    link.download = `facescan_${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();

    showToast('Foto capturada e salva com sucesso!', 'success');
  }

  // =========================================================================
  // Notificações Toast e Utilitários
  // =========================================================================
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '⚠️';

    toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3800);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, m => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[m]);
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================
  function bindEvents() {
    // Câmera
    dom.toggleCameraBtn.addEventListener('click', toggleCamera);
    dom.startCamPromptBtn.addEventListener('click', startCamera);

    // Troca de Câmera
    dom.cameraSelect.addEventListener('change', (e) => {
      state.deviceId = e.target.value;
      if (state.isCameraRunning) {
        stopCamera();
        startCamera();
      }
    });

    // Inverter / Modo Espelho
    dom.flipCameraBtn.addEventListener('click', () => {
      state.mirrorMode = !state.mirrorMode;
      dom.viewportContainer.classList.toggle('mirror-mode', state.mirrorMode);
      showToast(state.mirrorMode ? 'Modo Espelho ativado' : 'Modo Espelho desativado', 'info');
    });

    // Slider de Sensibilidade
    dom.thresholdSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.threshold = val / 100;
      dom.thresholdValue.textContent = `${val}%`;
    });

    // Toggles Visuais
    dom.toggleMesh.addEventListener('change', (e) => state.showMesh = e.target.checked);
    dom.toggleBox.addEventListener('change', (e) => state.showBox = e.target.checked);
    dom.togglePoints.addEventListener('change', (e) => state.showPoints = e.target.checked);
    dom.toggleScanner.addEventListener('change', (e) => {
      state.showScanner = e.target.checked;
      if (state.isCameraRunning && state.showScanner) {
        dom.laserScanner.classList.add('active');
      } else {
        dom.laserScanner.classList.remove('active');
      }
    });

    // Cadastro
    dom.openRegisterModalBtn.addEventListener('click', openRegisterModal);
    dom.closeRegisterModalBtn.addEventListener('click', closeRegisterModal);
    dom.cancelRegisterBtn.addEventListener('click', closeRegisterModal);
    dom.savePersonBtn.addEventListener('click', saveRegisteredPerson);

    // Submissão com Enter no input
    dom.personNameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        saveRegisteredPerson();
      }
    });

    // Fechar modal clicando fora
    dom.registerModal.addEventListener('click', (e) => {
      if (e.target === dom.registerModal) closeRegisterModal();
    });

    // Captura de Foto
    dom.takeSnapshotBtn.addEventListener('click', takeSnapshot);

    // Limpar Todos
    dom.clearAllFacesBtn.addEventListener('click', clearAllFaces);
  }

  // =========================================================================
  // Inicialização Geral da Aplicação
  // =========================================================================
  async function init() {
    bindEvents();
    await populateCameraDevices();
    await loadFaces();
    await initMediaPipe();
  }

  // Inicia ao carregar a página
  window.addEventListener('DOMContentLoaded', init);

})();
