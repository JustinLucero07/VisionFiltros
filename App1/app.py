# Author: vlarobbyk (Optimized Version)
# Version: 2.3 - Optimizado para performance
# Date: 2025-11-17

from flask import Flask, render_template, Response, request
import time
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Configuración del stream
_URL = 'http://192.168.0.106'
_PORT = '81'
_ST = '/stream'
stream_url = f'{_URL}:{_PORT}{_ST}'

# Variables globales
prev_time = time.time()
fps = 0.0
REFERENCE_WIDTH = 320
REFERENCE_HEIGHT = 240

# Cache de kernels (precalculados)
KERNEL_CACHE = {}
MORPHOLOGY_KERNEL = np.ones((3,3), np.uint8)

def get_gaussian_kernel(kernel_size, sigma, device='cpu'):
    """Cache de kernels gaussianos para PyTorch"""
    key = (kernel_size, sigma, device)
    if key not in KERNEL_CACHE:
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        ax = np.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        
        kernel_t = torch.from_numpy(kernel.astype(np.float32))
        weight = kernel_t.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1).to(device)
        KERNEL_CACHE[key] = weight
    
    return KERNEL_CACHE[key]


def calcular_fondo():
    """Calcula fondo con mediana de 30 frames"""
    global REFERENCE_WIDTH, REFERENCE_HEIGHT
    print("Calculando fondo...")
    
    res = requests.get(stream_url, stream=True)
    bytes_buffer = b''
    bg_frames = []
    
    while len(bg_frames) < 30:
        for chunk in res.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1 and b > a:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    if REFERENCE_WIDTH is None or REFERENCE_HEIGHT is None:
                        REFERENCE_HEIGHT, REFERENCE_WIDTH = frame.shape[:2]
                        print(f"Dimensiones: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
                    
                    if frame.shape[:2] != (REFERENCE_HEIGHT, REFERENCE_WIDTH):
                        frame = cv2.resize(frame, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                    
                    bg_frames.append(frame)
                    break
    
    bg = np.median(bg_frames, axis=0).astype(np.uint8)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)


def procesar_frame_optimizado(cv_img, bg_median_gray, params):
    """Pipeline de procesamiento optimizado"""
    h, w = REFERENCE_HEIGHT, REFERENCE_WIDTH
    
    # Conversión única a escala de grises
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # === RUIDO BÁSICO ===
    noise_mask = np.zeros((h, w), dtype=np.uint8)
    N = 537
    noise_mask[np.random.randint(0, h, N), np.random.randint(0, w, N)] = 255
    noise_image = cv2.bitwise_or(gray, noise_mask)
    
    # === ELIMINACIÓN DE FONDO ===
    if bg_median_gray.shape != (h, w):
        bg_resized = cv2.resize(bg_median_gray, (w, h))
    else:
        bg_resized = bg_median_gray
    
    diff = cv2.absdiff(gray, bg_resized)
    _, fondo_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    fondo_mask = cv2.morphologyEx(fondo_mask, cv2.MORPH_OPEN, MORPHOLOGY_KERNEL)
    
    # === MEJORAS DE ILUMINACIÓN ===
    hist_eq = cv2.equalizeHist(gray)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe_obj.apply(gray)
    
    # Retinex optimizado
    img_f = gray.astype(np.float32) + 1.0
    log_blur = cv2.GaussianBlur(img_f, (15,15), 30)
    retinex = np.log(img_f) - np.log(log_blur + 1.0)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # === OBJETO EXTRAÍDO ===
    mask_3ch = cv2.cvtColor(fondo_mask, cv2.COLOR_GRAY2BGR)
    solo_objeto = cv2.bitwise_and(cv_img, mask_3ch)
    
    # === RUIDO GAUSSIANO Y SPECKLE ===
    noisy_color = cv_img.copy()
    if params['gauss_prob'] > 0:
        gauss = np.random.normal(params['gauss_mean'], params['gauss_sigma'], cv_img.shape).astype(np.float32)
        mask = (np.random.rand(h, w, 1) < params['gauss_prob']).astype(np.float32)
        noisy_color = np.clip(noisy_color.astype(np.float32) + gauss * mask, 0, 255).astype(np.uint8)
    
    if params['speck_prob'] > 0:
        gauss = np.random.normal(0, np.sqrt(params['speck_var']), cv_img.shape).astype(np.float32)
        mask = (np.random.rand(h, w, 1) < params['speck_prob']).astype(np.float32)
        img_f = noisy_color.astype(np.float32)
        noisy_color = np.clip(img_f + img_f * gauss * mask, 0, 255).astype(np.uint8)
    
    # === DENOISING PYTORCH ===
    weight = get_gaussian_kernel(params['pytorch_kernel'], params['pytorch_sigma'], 'cpu')
    img_t = torch.from_numpy(noisy_color.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    pad = params['pytorch_kernel'] // 2
    img_t = F.pad(img_t, (pad, pad, pad, pad), mode='reflect')
    denoised_t = F.conv2d(img_t, weight, stride=1, padding=0, groups=3)
    denoised_torch = np.clip(denoised_t.squeeze(0).numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    
    # === COMPARACIÓN ===
    diff_mask = cv2.cvtColor(cv2.absdiff(cv_img, denoised_torch), cv2.COLOR_BGR2GRAY)
    _, diff_mask_bin = cv2.threshold(diff_mask, 20, 255, cv2.THRESH_BINARY)
    comparison = np.hstack([cv_img, cv_img.copy()])
    mask_bool = diff_mask_bin.astype(bool)
    for ch in range(3):
        comparison[:, w:, ch][mask_bool] = denoised_torch[:, :, ch][mask_bool]
    
    # === FILTROS ESPACIALES (solo los necesarios para visualización) ===
    spatial = {
        'median_3': cv2.medianBlur(noisy_color, 3),
        'blur_3': cv2.blur(noisy_color, (3, 3)),
        'gaussian_3': cv2.GaussianBlur(noisy_color, (3, 3), 0),
        'median_5': cv2.medianBlur(noisy_color, 5),
        'blur_5': cv2.blur(noisy_color, (5, 5)),
        'gaussian_5': cv2.GaussianBlur(noisy_color, (5, 5), 0),
        'median_7': cv2.medianBlur(noisy_color, 7),
        'blur_7': cv2.blur(noisy_color, (7, 7)),
        'gaussian_7': cv2.GaussianBlur(noisy_color, (7, 7), 0)
    }
    
    # === DETECCIÓN DE BORDES ===
    gray_noisy = cv2.cvtColor(noisy_color, cv2.COLOR_BGR2GRAY)
    canny_raw = cv2.Canny(gray_noisy, 50, 150)
    sobelx = cv2.Sobel(gray_noisy, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_noisy, cv2.CV_64F, 0, 1, ksize=3)
    sobel_raw = np.clip(cv2.magnitude(sobelx, sobely) / 4, 0, 255).astype(np.uint8)
    
    # Con suavizado
    gray_median = cv2.medianBlur(gray_noisy, 3)
    canny_med = cv2.Canny(gray_median, 50, 150)
    sobelx_m = cv2.Sobel(gray_median, cv2.CV_64F, 1, 0, ksize=3)
    sobely_m = cv2.Sobel(gray_median, cv2.CV_64F, 0, 1, ksize=3)
    sobel_med = np.clip(cv2.magnitude(sobelx_m, sobely_m) / 4, 0, 255).astype(np.uint8)
    
    gray_gauss = cv2.GaussianBlur(gray_noisy, (5, 5), 0)
    canny_gauss = cv2.Canny(gray_gauss, 50, 150)
    sobelx_g = cv2.Sobel(gray_gauss, cv2.CV_64F, 1, 0, ksize=3)
    sobely_g = cv2.Sobel(gray_gauss, cv2.CV_64F, 0, 1, ksize=3)
    sobel_gauss = np.clip(cv2.magnitude(sobelx_g, sobely_g) / 4, 0, 255).astype(np.uint8)
    
    # Conversiones optimizadas a BGR (solo una vez)
    to_bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    
    return {
        'gray': gray, 'noise': noise_image, 'conv': noise_image,
        'fondo_mask': fondo_mask, 'objeto': solo_objeto,
        'hist_eq': hist_eq, 'clahe': clahe, 'retinex': retinex,
        'noisy_color': noisy_color, 'denoised': denoised_torch,
        'comparison': comparison, 'spatial': spatial,
        'canny_raw': canny_raw, 'sobel_raw': sobel_raw,
        'canny_med': canny_med, 'sobel_med': sobel_med,
        'canny_gauss': canny_gauss, 'sobel_gauss': sobel_gauss,
        'to_bgr': to_bgr
    }


def crear_canvas_optimizado(cv_img, results, fps_text):
    """Crea el canvas de visualización de forma optimizada"""
    h, w = REFERENCE_HEIGHT, REFERENCE_WIDTH
    COLS = 4
    TITLE_HEIGHT = 30
    
    # Lista de imágenes (evitamos conversiones redundantes)
    to_bgr = results['to_bgr']
    images = [
        ("ORIGINAL", cv_img),
        ("GRAY", to_bgr(results['gray'])),
        ("RUIDO", to_bgr(results['noise'])),
        ("CONV", to_bgr(results['conv'])),
        
        ("FONDO MASK", to_bgr(results['fondo_mask'])),
        ("OBJETO", results['objeto']),
        ("NOISY COLOR", results['noisy_color']),
        ("PT DENOISED", results['denoised']),
        
        ("HIST EQ", to_bgr(results['hist_eq'])),
        ("CLAHE", to_bgr(results['clahe'])),
        ("RETINEX", to_bgr(results['retinex'])),
        ("COMP ORIG", results['comparison'][:, :w]),
        
        ("COMP DENOISE", results['comparison'][:, w:]),
        ("MEDIAN 3", results['spatial']['median_3']),
        ("BLUR 3", results['spatial']['blur_3']),
        ("GAUSSIAN 3", results['spatial']['gaussian_3']),
        
        ("MEDIAN 5", results['spatial']['median_5']),
        ("BLUR 5", results['spatial']['blur_5']),
        ("GAUSSIAN 5", results['spatial']['gaussian_5']),
        ("MEDIAN 7", results['spatial']['median_7']),
        
        ("BLUR 7", results['spatial']['blur_7']),
        ("GAUSSIAN 7", results['spatial']['gaussian_7']),
        ("CANNY RAW", to_bgr(results['canny_raw'])),
        ("SOBEL RAW", to_bgr(results['sobel_raw'])),
        
        ("CANNY MED3", to_bgr(results['canny_med'])),
        ("SOBEL MED3", to_bgr(results['sobel_med'])),
        ("CANNY GAUSS5", to_bgr(results['canny_gauss'])),
        ("SOBEL GAUSS5", to_bgr(results['sobel_gauss']))
    ]
    
    rows = (len(images) + COLS - 1) // COLS
    canvas = np.zeros((rows * (h + TITLE_HEIGHT), COLS * w, 3), dtype=np.uint8)
    
    # Llenar canvas (optimizado)
    for idx, (title, img) in enumerate(images):
        row, col = divmod(idx, COLS)
        y = row * (h + TITLE_HEIGHT)
        x = col * w
        
        # Colocar imagen directamente (ya tienen el tamaño correcto)
        canvas[y + TITLE_HEIGHT:y + TITLE_HEIGHT + h, x:x + w] = img
        
        # Título
        cv2.putText(canvas, title, (x + 10, y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # FPS
    cv2.putText(canvas, fps_text, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return canvas


def video_capture():
    """Generador principal del stream de video"""
    global prev_time, fps
    
    # Parámetros de la interfaz
    params = {
        'gauss_mean': float(request.args.get('gauss_mean', 0.0)),
        'gauss_sigma': float(request.args.get('gauss_sigma', 20.0)),
        'gauss_prob': float(request.args.get('gauss_prob', 0.0)),
        'speck_var': float(request.args.get('speck_var', 0.04)),
        'speck_prob': float(request.args.get('speck_prob', 0.0)),
        'pytorch_kernel': int(request.args.get('pt_k', 5)),
        'pytorch_sigma': float(request.args.get('pt_sigma', 1.2))
    }
    
    bg_median_gray = calcular_fondo()
    res = requests.get(stream_url, stream=True)
    bytes_buffer = b''
    
    for chunk in res.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1 and b > a:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            
            cv_img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if cv_img is None:
                continue
            
            try:
                # Redimensionar si es necesario
                if cv_img.shape[:2] != (REFERENCE_HEIGHT, REFERENCE_WIDTH):
                    cv_img = cv2.resize(cv_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                
                # FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time + 1e-6)
                prev_time = current_time
                fps_text = f"FPS: {fps:.2f} | {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}"
                
                # Procesamiento
                results = procesar_frame_optimizado(cv_img, bg_median_gray, params)
                
                # Canvas
                canvas = crear_canvas_optimizado(cv_img, results, fps_text)
                
                # Encode y yield
                _, encoded = cv2.imencode(".jpg", canvas)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + 
                       encoded.tobytes() + b"\r\n")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False, port=5005, threaded=True)